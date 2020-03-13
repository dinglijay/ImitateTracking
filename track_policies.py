import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import batch_to_seq, conv_to_fc, seq_to_batch
from baselines.common.models import register, nature_cnn

import tensorflow.contrib.slim as slim
import joblib
import os
import gym

from baselines.common import tf_util as U

def load_vggm_conv(load_path, variables=None, sess=None):
    import joblib
    sess = sess or tf.get_default_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    for v in variables:
        if 'conv' in v.name and 'Adam' not in v.name:
            restores.append(v.assign(loaded_params[v.name[v.name.find('conv'):]]))
            print('Loaded: ', v.name)
    sess.run(restores)

def vggm1234(x, TRAIN_COVN=True):

    net = slim.convolution(x, 96, [7, 7], 2, padding='VALID', scope='conv1',
                            activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, trainable=TRAIN_COVN)
    net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*1, beta=0.75)
    net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool1')

    net = slim.convolution(net, 256, [5, 5], 2, padding='VALID', scope='conv2',
                            activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, trainable=TRAIN_COVN)
    net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*1, beta=0.75)
    net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool2')

    net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv3',
                            activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, trainable=TRAIN_COVN)
    
    net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv4',
                            activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, trainable=TRAIN_COVN)

    return U.flattenallbut0(net)


@register("track_2cnn_fc12")
def track_2cnn_fc12(**conv_kwargs):
    def network_fn(X, nenv=1):

        ob_g, ob_l = tf.split(X,2,axis=1)
        ob_g = tf.squeeze(ob_g,axis=1) - 128.0
        ob_l = tf.squeeze(ob_l,axis=1) - 128.0
        
        # Conv layer
        net_g = vggm1234(ob_g)
        net_l = vggm1234(ob_l)
        feat = tf.concat([net_g, net_l], 1)
          
        # fcs_actor            
        h = slim.fully_connected(feat, 512, scope='fc1', activation_fn=tf.nn.relu)
        # h = slim.fully_connected(h, 4, scope='fc2', activation_fn=None)
       
        return h

    return network_fn


@register("track_2cnn_lstm")
def track_2cnn_lstm(nlstm=256, layer_norm=False, conv_fn=nature_cnn, is_training = True, **conv_kwargs):
    def network_fn(X, nenv=1):

        nbatch = X.shape[0]
        nsteps = nbatch // nenv
        
        ob_g, ob_l = tf.split(X,2,axis=1)
        ob_g = tf.squeeze(ob_g,axis=1) - 128.0
        ob_l = tf.squeeze(ob_l,axis=1) - 128.0
        
        # Conv layer
        net_g = vggm1234(ob_g)
        net_l = vggm1234(ob_l)
        feat = tf.concat([net_g, net_l], 1)

        # LSTM
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(feat, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)
        
        return (feat, h), {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

@register("track_2cnn_lstm_fc")
def track_cnn_lstm_fc(nlstm=256, layer_norm=False, conv_fn=nature_cnn, is_training = True, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        ob_g, ob_l = tf.split(X,2,axis=1)
        ob_g = tf.squeeze(ob_g,axis=1) - 128.0
        ob_l = tf.squeeze(ob_l,axis=1) - 128.0
        
        # Conv layer
        net_g = vggm1234(ob_g)
        net_l = vggm1234(ob_l)
        feat = tf.concat([net_g, net_l], 1)

        # LSTM
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(feat, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)
        
        # FC
        h = slim.fully_connected(h, 4, scope='fc', activation_fn=tf.nn.tanh)
        
        return (feat, h), {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

@register("value_cnn_fc12")
def value_cnn_fc12(X):

    ob_g, ob_l = tf.split(X,2,axis=1)
    ob = tf.squeeze(ob_g,axis=1) - 128.0
    feat = vggm1234(ob)
    h = slim.fully_connected(feat, 256, scope='fc1', activation_fn=tf.nn.relu)
    h = slim.fully_connected(h, 1, scope='fc2')
       
    return h