from __future__ import division
from track_policies import vggm1234

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import joblib
import os
import logging

logger = logging.getLogger('networks')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class ADNetwork:

    def __init__(self, learning_rate=1e-04):
        self.input_tensor = None
        self.label_tensor = None
        self.class_tensor = None
        self.layer_feat = None
        self.layer_actions = None
        self.layer_scores = None

        self.loss_actions = None
        self.loss_cls = None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.weighted_grads_fc1 = None
        self.weighted_grads_fc2 = None
        self.var_grads_fc1 = None
        self.var_grads_fc2 = None
        self.weighted_grads_op1 = None
        self.weighted_grads_op2 = None

    def read_original_weights(self, tf_session=None, path='./log/0228_trackCnnFc12/checkpoints/01300'):

        tf_session = tf_session or tf.Session()
        tf_session.run(tf.global_variables_initializer())

        weights = joblib.load(path)
        tensors_weights = { 'conv1/weights:0'   : 'ppo2_model/pi/conv1/weights:0',
                            'conv1/biases:0'    : 'ppo2_model/pi/conv1/biases:0',
                            'conv2/weights:0'   : 'ppo2_model/pi/conv2/weights:0',
                            'conv2/biases:0'    : 'ppo2_model/pi/conv2/biases:0',
                            'conv3/weights:0'   : 'ppo2_model/pi/conv3/weights:0',
                            'conv3/biases:0'    : 'ppo2_model/pi/conv3/biases:0',
                            'conv4/weights:0'   : 'ppo2_model/pi/conv4/weights:0',
                            'conv4/biases:0'    : 'ppo2_model/pi/conv4/biases:0',
                            'polfc1/weights:0'  : 'ppo2_model/pi/fc1/weights:0',
                            'polfc1/biases:0'   : 'ppo2_model/pi/fc1/biases:0',
                            'polfc2/weights:0'  : 'ppo2_model/pi/w:0',
                            'polfc2/biases:0'   : 'ppo2_model/pi/b:0'
                            # 'vffc1/weights:0'   : 'pi/vffc1/weights:0',
                            # 'vffc1/biases:0'    : 'pi/vffc1/biases:0',
                            # 'vffc2/weights:0'   : 'pi/vffc2/weights:0',
                            # 'vffc2/biases:0'    : 'pi/vffc2/biases:0'
                           }
        for var in tf.trainable_variables():
            if 'vffc' not in var.name:
                val = weights[tensors_weights[var.name]]
                tf_session.run(var.assign(val))
                print('Original weights assigned: %s' % var.name)
        print(tf_session.run(tf.report_uninitialized_variables()))


  
    def create_network(self, input_tensor, label_tensor, class_tensor, is_training):
        
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.class_tensor = class_tensor

        # feature extractor - img_g
        ob = input_tensor - 128.0
        ob_g, ob_l = tf.split(ob,2,axis=1)
        ob_g = tf.squeeze(ob_g,axis=1)
        ob_l = tf.squeeze(ob_l,axis=1)

        # Conv layer
        net_g = vggm1234(ob_g)
        net_l = vggm1234(ob_l)
        self.layer_feat = feat = tf.concat([net_g, net_l], 1)
          
        # fcs_actor            
        net = slim.fully_connected(feat, 512, scope='polfc1', activation_fn=tf.nn.relu)
        out_actions = slim.fully_connected(net, 4, scope='polfc2', activation_fn=None)

        # vfs_value
        net = slim.fully_connected(feat, 512, scope='vffc1', activation_fn=tf.nn.relu) 
        out_scores = slim.fully_connected(net, 2, scope='vffc2', activation_fn=None) 
        
        # auxilaries
        self.layer_actions = out_actions
        self.layer_scores = tf.nn.softmax(out_scores)

        # losses
        self.loss_actions = tf.reduce_mean(tf.squared_difference(label_tensor, out_actions))
        self.loss_cls = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=class_tensor, logits=out_scores))

        # finetune ops
        self.var_grads_fc1 = var_fc = [var for var in tf.trainable_variables() if 'polfc' in var.name]
        gradients1 = tf.gradients(self.loss_actions, xs=var_fc)      # only finetune on fc layers of actor
        self.weighted_grads_fc1 = []
        for var, grad in zip(var_fc, gradients1):
            self.weighted_grads_fc1.append(10 * grad)

        self.var_grads_fc2 = var_fc = [var for var in tf.trainable_variables() if 'vffc' in var.name]
        gradients2 = tf.gradients(self.loss_cls, xs=var_fc)          # only finetune on fc layers pf value_fn
        self.weighted_grads_fc2 = []
        for var, grad in zip(var_fc, gradients2):
            self.weighted_grads_fc2.append(10 * grad)

        self.weighted_grads_op1 = self.optimizer.apply_gradients(zip(self.weighted_grads_fc1, self.var_grads_fc1))
        self.weighted_grads_op2 = self.optimizer.apply_gradients(zip(self.weighted_grads_fc2, self.var_grads_fc2))


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from conf.configs import ADNetConf
    ADNetConf.get('./conf/dylan.yaml')
    
    input_node = tf.placeholder(tf.float32, shape=(None, 2, 107, 107, 3), name='patch')
    tensor_lb_action = tf.placeholder(tf.float32, shape=(None, 4), name='lb_action')    # actions
    tensor_lb_class = tf.placeholder(tf.int32, shape=(None, ), name='lb_class')      # 2 actions
    is_training = tf.placeholder(tf.bool, name='is_training')

    adnet = ADNetwork()
    adnet.create_network(input_node, tensor_lb_action, tensor_lb_class, is_training)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        # load all pretrained weights
        adnet.read_original_weights(sess)
        
        variables = tf.trainable_variables()
        ps = sess.run(variables)

        # zero input
        zeros = np.zeros(shape=(1, 2, 107, 107, 3), dtype=np.float32)
        from commons import imread
        img1 = imread(r'./dataset/OTB/Car4/img/0009.jpg')
        from boundingbox import crop_resize
        zeros = crop_resize(img1, (70, 51, 103, 87))[None]
        

        action_out, class_out = sess.run([adnet.layer_actions, adnet.layer_scores], feed_dict={input_node: zeros})
        print(action_out, class_out)
