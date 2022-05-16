import numpy as np
import tensorflow as tf
from ops import linear
from ops import deconv2d
from util import log
from ops import non_local_block, conditional_batch_norm
import time

def conv_out_size_same(size, stride):
    return int(np.ceil(float(size) / float(stride)))
  
class Generator(object):
    def __init__(self, name, h, w, c, is_train, use_sn):
    
        self.name = name
        self.s_h, self.s_w, self.colors = [h,w,c]
        self.s_h2, self.s_w2 = conv_out_size_same(self.s_h, 2), conv_out_size_same(self.s_w, 2)
        self.s_h4, self.s_w4 = conv_out_size_same(self.s_h2, 2), conv_out_size_same(self.s_w2, 2)
        self.s_h8, self.s_w8 = conv_out_size_same(self.s_h4, 2), conv_out_size_same(self.s_w4, 2)
        self._is_train = is_train
        self.use_sn = use_sn
        self._embed_y_dim = 128
        
    def __call__(self, z, y):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            #因为生成器根本不在测试阶段使用，所以batch_norm操作不需要分训练和测试
            batch_size = z.shape[0]
            
            y = tf.concat([z, y], axis=1)
            z = y
            
            net = linear(z, self.s_h8 * self.s_w8 * 512, scope="g_fc1", use_sn=self.use_sn)
            net = tf.reshape(net, [batch_size, self.s_h8, self.s_w8, 512])
            net = conditional_batch_norm(net, y, is_training=self._is_train, use_sn = self.use_sn, center=True, scale=True, name="g_cbn_deconv0", use_bias=False)
            net = tf.nn.relu(net)
            
            net = deconv2d(net, [batch_size, self.s_h4, self.s_w4, 256], 4, 4, 2, 2, name="g_dc1", use_sn=self.use_sn)
            net = conditional_batch_norm(net, y, is_training=self._is_train, use_sn = self.use_sn, center=True, scale=True, name="g_cbn_deconv1", use_bias=False)
            net = tf.nn.relu(net)
            
            #net = non_local_block(net, name='self_attention_generator', use_sn=self.use_sn)
            
            net = deconv2d(net, [batch_size, self.s_h2, self.s_w2, 128], 4, 4, 2, 2, name="g_dc2", use_sn=self.use_sn)
            net = conditional_batch_norm(net, y, is_training=self._is_train, use_sn = self.use_sn, center=True, scale=True, name="g_cbn_deconv2", use_bias=False)
            net = tf.nn.relu(net)
            
            net = deconv2d(net, [batch_size, self.s_h, self.s_w, 64], 4, 4, 2, 2, name="g_dc3", use_sn=self.use_sn)
            net = tf.contrib.layers.batch_norm(net,
                    updates_collections=None, is_training=self._is_train, center=True, scale=True, decay=0.9, epsilon=1e-5, scope="g_cbn_deconv3")
            net = tf.nn.relu(net)
            
            net = deconv2d(net, [batch_size, self.s_h, self.s_w, self.colors], 3, 3, 1, 1, name="g_dc4", use_sn=self.use_sn)
            out = tf.div(tf.nn.tanh(net) + 1.0, 2.0)
            
            return out
