'''
from baselines/gail/mlp_policy.py and add simple modification
(1) a vgg-m network
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import joblib
import os
import gym
import numpy as np

from baselines.common import tf_util as U
from baselines.common.distributions import make_pdtype

from conf.configs import ADNetConf
from track_policies import vggm1234

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TrackPolicy(object):
    recurrent = False
    
    def __init__(self, name, load_path=None, reuse=False, *args, **kwargs):
        
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.scope = tf.get_variable_scope().name
            self.__build_graph(*args, **kwargs)
        print(tf.trainable_variables())
        self.read_original_weights(load_path)

    def __build_graph(self, ob_space, ac_space, gaussian_fixed_var=True):
        
        self.pdtype = pdtype = make_pdtype(ac_space)
        
        assert not isinstance(ob_space, gym.spaces.tuple.Tuple)
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        ob_g, ob_l = tf.split(ob,2,axis=1)
        ob_g = tf.squeeze(ob_g,axis=1) - 128.0
        ob_l = tf.squeeze(ob_l,axis=1) - 128.0

        # Conv layer
        net = slim.convolution(ob_g, 96, [7, 7], 2, padding='VALID', scope='conv1',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*1, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool1')

        net = slim.convolution(net, 256, [5, 5], 2, padding='VALID', scope='conv2',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*1, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool2')

        net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv3',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)
        
        net_g = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv4',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)

        net = slim.convolution(ob_l, 96, [7, 7], 2, padding='VALID', scope='conv1',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*1, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool1')

        net = slim.convolution(net, 256, [5, 5], 2, padding='VALID', scope='conv2',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*1, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool2')

        net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv3',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)
        
        net_l = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv4',
                               activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE)

        # Concat Features
        self.feat = feat = tf.concat([U.flattenallbut0(net_g), U.flattenallbut0(net_l)], 1)
          
        # fcs_actor            
        net = slim.fully_connected(feat, 512, scope='polfc1', activation_fn=tf.nn.relu)
        # pdparam = slim.fully_connected(net, 4, scope='polfc2', activation_fn=None)
        mean = slim.fully_connected(net, pdtype.param_shape()[0]//2, 
                                    scope='polfc2', activation_fn=None)
        logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], 
                                     initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, logstd], axis=1)
        self.pd = pdtype.pdfromflat(pdparam)

        # fcs_value
        net = slim.fully_connected(feat, 512, scope='vffc1', activation_fn=tf.nn.relu) 
        self.vpred = slim.fully_connected(net, 1, scope='vffc2', activation_fn=None) 

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1, vpred1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
    
    def read_original_weights(self, load_path):
        U.initialize()
        tf_session = tf.get_default_session()

        if load_path == None:
            load_path = './log/0228_trackCnnFc12/checkpoints/01300'

        weights = joblib.load(load_path)
        tensors_weights = { self.scope+'/conv1/weights:0'   : 'ppo2_model/pi/conv1/weights:0',
                            self.scope+'/conv1/biases:0'    : 'ppo2_model/pi/conv1/biases:0',
                            self.scope+'/conv2/weights:0'   : 'ppo2_model/pi/conv2/weights:0',
                            self.scope+'/conv2/biases:0'    : 'ppo2_model/pi/conv2/biases:0',
                            self.scope+'/conv3/weights:0'   : 'ppo2_model/pi/conv3/weights:0',
                            self.scope+'/conv3/biases:0'    : 'ppo2_model/pi/conv3/biases:0',
                            self.scope+'/conv4/weights:0'   : 'ppo2_model/pi/conv4/weights:0',
                            self.scope+'/conv4/biases:0'    : 'ppo2_model/pi/conv4/biases:0',
                            self.scope+'/polfc1/weights:0'  : 'ppo2_model/pi/fc1/weights:0',
                            self.scope+'/polfc1/biases:0'   : 'ppo2_model/pi/fc1/biases:0',
                            self.scope+'/polfc2/weights:0'  : 'ppo2_model/pi/w:0',
                            self.scope+'/polfc2/biases:0'   : 'ppo2_model/pi/b:0'
                            # 'vffc1/weights:0'   : 'pi/vffc1/weights:0',
                            # 'vffc1/biases:0'    : 'pi/vffc1/biases:0',
                            # 'vffc2/weights:0'   : 'pi/vffc2/weights:0',
                            # 'vffc2/biases:0'    : 'pi/vffc2/biases:0'
                           }
        for var in tf.trainable_variables():
            if 'vffc' not in var.name and 'logstd' not in var.name:
                val = weights[tensors_weights[var.name]]
                tf_session.run(var.assign(val))
                print('Original weights assigned: %s' % var.name)
        print(tf_session.run(tf.report_uninitialized_variables()))    


class ADNetwork(object):
    NUM_ACTIONS = 11
    NUM_ACTION_HISTORY = 10
    ACTION_IDX_STOP = 8

    def __init__(self, learning_rate=1e-04):
        self.input_tensor = None
        self.label_tensor = None
        self.class_tensor = None
        self.action_history_tensor = None
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

    def create_network(self, input_tensor, label_tensor, class_tensor, action_history_tensor, is_training):
        
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.class_tensor = class_tensor
        self.action_history_tensor = action_history_tensor

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
        h = slim.fully_connected(feat, 512, scope='polfc1', activation_fn=tf.nn.relu)
        out_actions = 0.3 * slim.fully_connected(h, 4, scope='polfc2', activation_fn=None)

        # fcs_value
        # net = slim.fully_connected(feat, 512, scope='vffc1', activation_fn=tf.nn.relu) 
        out_scores = slim.fully_connected(h, 2, scope='vffc1', activation_fn=None) 
        
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

    def read_original_weights(self, tf_session=None, load_path='./log/0309_track2CnnFc12_noAct/checkpoints/01100'):

        tf_session = tf_session or tf.Session()
        tf_session.run(tf.global_variables_initializer())
        weights = joblib.load(load_path)
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
                            'polfc2/biases:0'   : 'ppo2_model/pi/b:0'}

        for var in tf.trainable_variables():
            if 'vffc' not in var.name and 'logstd' not in var.name:
                val = weights[tensors_weights[var.name]]
                tf_session.run(var.assign(val))
                print('Original weights assigned: %s' % var.name)
        print(tf_session.run(tf.report_uninitialized_variables()))

def show_variables( variables=None, sess=None):
    sess = sess or tf.get_default_session()
    variables = variables or tf.trainable_variables()

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    return save_dict
    
if __name__ == '__main__':
    # from PIL import Image
    # from boundingbox import BoundingBox, crop_resize

    # if 'track-v0' in gym.envs.registry.env_specs:
    #     print('Remove track-v0 from registry')
    #     del gym.envs.registry.env_specs['track-v0']
    # import track_env
    # env = gym.make('track-v0')
    ADNetConf.get('conf/dylan.yaml')
    
    # img1 = Image.open(r'0028.jpg', mode='r')  
    # bbox = BoundingBox(331,152,26,61)
    # img_gl = crop_resize(img1, bbox)

    # pi = TrackPolicy("pi", ob_space=env.observation_space, ac_space=env.action_space)
        
    # ac1, vpred1 = pi.act(stochastic=False, ob=img_gl)
    # print(ac1)
    # print(vpred1)
    input_node = tf.placeholder(tf.float32, shape=(None, 2, 107, 107, 3), name='patch')
    tensor_lb_action = tf.placeholder(tf.float32, shape=(None, 4), name='lb_action')    # 11 actions
    tensor_lb_class = tf.placeholder(tf.int32, shape=(None, ), name='lb_class')      # 2 actions
    action_history_tensor = tf.placeholder(tf.float32, shape=(None, 1, 1, ADNetwork.NUM_ACTIONS * ADNetwork.NUM_ACTION_HISTORY), name='action_history')
    is_training = tf.placeholder(tf.bool, name='is_training')

    adnet = ADNetwork()
    adnet.create_network(input_node, tensor_lb_action, tensor_lb_class, action_history_tensor, is_training)

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
        pass
    
