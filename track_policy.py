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

from baselines.common import tf_util as U
from baselines.common.distributions import make_pdtype

from conf.configs import ADNetConf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TrackPolicy(object):
    recurrent = False
    
    def __init__(self, name, reuse=False, *args, **kwargs):
        
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.scope = tf.get_variable_scope().name
            self.__build_graph(*args, **kwargs)
            self.read_original_weights()

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
        out_limit = ADNetConf.g()['dl_paras']['actor_out_limt']
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = out_limit * slim.fully_connected(net, pdtype.param_shape()[0]//2, 
                                        scope='polfinal', activation_fn=tf.nn.tanh)
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], 
                                     initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, logstd], axis=1)
        else:
            pdparam = out_limit * slim.fully_connected(net, pdtype.param_shape()[0], 
                                           scope='polfinal', activation_fn=tf.nn.tanh)

        # fcs_value
        net = slim.fully_connected(feat, 512, scope='vffc1', activation_fn=tf.nn.relu) 
        self.vpred = slim.fully_connected(net, 1, scope='vffc2', activation_fn=tf.nn.tanh) 

                        
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

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
    
    def read_original_weights(self):
        U.initialize()
        tf_session = tf.get_default_session()

        weights = joblib.load(os.path.expanduser('./checkpoint/ACT1-4.ckpt'))
#        weights_pol = joblib.load(os.path.expanduser('./checkpoint/no_fineture.reword_limitation_0.3/Actor_Disc004000.ckpt'))
#        weights.update(weights_pol)
        tensors_weights = { self.scope+'/conv1/weights:0'   : 'conv1/weights:0',
                            self.scope+'/conv1/biases:0'    : 'conv1/biases:0',
                            self.scope+'/conv2/weights:0'   : 'conv2/weights:0',
                            self.scope+'/conv2/biases:0'    : 'conv2/biases:0',
                            self.scope+'/conv3/weights:0'   : 'conv3/weights:0',
                            self.scope+'/conv3/biases:0'    : 'conv3/biases:0',
                            self.scope+'/conv4/weights:0'   : 'conv4/weights:0',
                            self.scope+'/conv4/biases:0'    : 'conv4/biases:0'
#                            self.scope+'/polfc1/weights:0'  : 'pi/polfc1/w:0',
#                            self.scope+'/polfc1/biases:0'   : 'pi/polfc1/b:0',
#                            self.scope+'/polfinal/weights:0': 'pi/polfinal/w:0',
#                            self.scope+'/polfinal/biases:0' : 'pi/polfinal/b:0',
#                            self.scope+'/logstd:0'          : 'pi/logstd:0',
#                            self.scope+'/vffc1/weights:0'   : 'pi/vffc1/w:0',
#                            self.scope+'/vffc1/biases:0'    : 'pi/vffc1/b:0',
#                            self.scope+'/vffc2/weights:0'   : 'pi/vffc2/w:0',
#                            self.scope+'/vffc2/biases:0'    : 'pi/vffc2/b:0'
                           }
        for var in self.get_trainable_variables():
            if 'conv' in var.name:
                val = weights[tensors_weights[var.name]]
                tf_session.run(var.assign(val))
                print('Original weights assigned: %s' % var.name)
    
#        print(tf_session.run(tf.report_uninitialized_variables()))
    

def show_variables( variables=None, sess=None):
    sess = sess or tf.get_default_session()
    variables = variables or tf.trainable_variables()

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    return save_dict
    
if __name__ == '__main__':
    from PIL import Image
    from boundingbox import BoundingBox, crop_resize

    if 'track-v0' in gym.envs.registry.env_specs:
        print('Remove track-v0 from registry')
        del gym.envs.registry.env_specs['track-v0']
    import track_env
    env = gym.make('track-v0')
    ADNetConf.get('conf/dylan.yaml')
    
    img1 = Image.open(r'0028.jpg', mode='r')  
    bbox = BoundingBox(331,152,26,61)
    img_gl = crop_resize(img1, bbox)

    pi = TrackPolicy("pi", ob_space=env.observation_space, ac_space=env.action_space)
        
    ac1, vpred1 = pi.act(stochastic=False, ob=img_gl)
    print(ac1)
    print(vpred1)
    
#    sess = tf.get_default_session()
#    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
#    import joblib
#    import os
#    weight = joblib.load(os.path.expanduser('checkpoint/Actor250000_from_ACT_acspace4.ckpt'))
#    
##    U.load_variables('checkpoint/Actor250000_from_ACT_acspace4.ckpt', variables=pi.get_variables())
#    
#    zeros = np.zeros(shape=(107,107, 3), dtype=np.int)
##    img_gl = np.array([zeros, zeros])
#    ac2, vpred2 = pi.act(stochastic=False, ob=img_gl)
#    print(ac1)
#    
