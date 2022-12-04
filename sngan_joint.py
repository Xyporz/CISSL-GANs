import numpy as np
import tensorflow as tf
from ops import conv2d
from ops import linear
from util import log
from ops import lrelu
from ops import non_local_block, spectral_norm
import time
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import cv2

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return tf.div(x + tf.constant(1e-10), tf.sqrt(tf.reduce_mean(tf.square(x))) + tf.constant(1e-10))

class Classifier_proD(object):
    def __init__(self, name, num_class, use_sn):
    
        self.name = name
        self._num_class = num_class
        self.use_sn = use_sn

    def __call__(self, input, y):
    
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
        
            input = input * 2.0 - 1.0
            
            net = conv2d(input, 64, 3, 3, 1, 1, name="d_conv1", use_sn=self.use_sn)
            net = lrelu(net, leak=0.1)
                
            net = conv2d(net, 128, 4, 4, 2, 2, name="d_conv2", use_sn=self.use_sn)
            net = lrelu(net, leak=0.1)
            
            net = conv2d(net, 128, 3, 3, 1, 1, name="d_conv3", use_sn=self.use_sn)
            net = lrelu(net, leak=0.1)
             
            net = conv2d(net, 256, 4, 4, 2, 2, name="d_conv4", use_sn=self.use_sn)
            net = lrelu(net, leak=0.1)
            
            net = conv2d(net, 256, 3, 3, 1, 1, name="d_conv5", use_sn=self.use_sn)
            net = lrelu(net, leak=0.1)
            
            net = conv2d(net, 512, 4, 4, 2, 2, name="d_conv6", use_sn=self.use_sn)
            net = lrelu(net, leak=0.1)
            
            net = conv2d(net, 512, 3, 3, 1, 1, name="d_conv7", use_sn=self.use_sn)
            net_conv = lrelu(net, leak=0.1)
            
            h = tf.layers.flatten(net_conv)
            out_logit = linear(h, self._num_class, scope="d_fc1", use_sn=self.use_sn)
            out_logit_tf = linear(h, 1, scope="final_fc", use_sn=self.use_sn)
            
            feature_matching = h
            
            log.info("[Discriminator] after final processing: %s", net_conv.shape)
            with tf.variable_scope("embedding_fc", reuse=tf.AUTO_REUSE):
                # We do not use ops.linear() below since it does not have an option to
                # override the initializer.
                kernel = tf.get_variable(
                    "kernel", [y.shape[1], h.shape[1]], tf.float32,
                    initializer=tf.initializers.glorot_normal())
                if self.use_sn:
                    kernel = spectral_norm(kernel)
                embedded_y = tf.matmul(y, kernel)
                out_logit_tf += tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
            
            return tf.nn.softmax(out_logit), out_logit, tf.nn.sigmoid(out_logit_tf), out_logit_tf, feature_matching
