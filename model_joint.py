from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from util import log
from sngan_gen import Generator
from sngan_gen_y import Generator as gGenerator
from sngan_joint import  Classifier_proD
import os
import time
import numpy as np
from diffAugment import DiffAugment

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size_G = self.config["batch_size_G"]
        self.batch_size_L = self.config["batch_size_L"]
        self.batch_size_U = self.config["batch_size_U"]
        
        self.h = self.config["h"]
        self.w = self.config["w"]
        self.c = self.config["c"]
        self.IMAGE_DIM = [self.h, self.w, self.c]
        
        self.len = self.config["len"]
        self.num_class = self.config["num_class"]
        self.n_z = self.config["n_z"]
        
        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')
        
        self.z_g_ph = tf.placeholder(tf.float32, [self.batch_size_G, self.n_z], name = 'latent_variable')  # latent variable
        self.z_g_ph_linspace = tf.placeholder(tf.float32, [10, self.n_z], name = 'latent_variable_linspace')
        self.y_g_ph = tf.placeholder(tf.float32, [self.batch_size_G, self.num_class], name='condition_label')
        
        self.x_l_ph = tf.placeholder(tf.float32, [self.batch_size_L] + self.IMAGE_DIM, name='labeled_images')
        self.y_l_ph = tf.placeholder(tf.float32, [self.batch_size_L, self.num_class], name='real_label')
        
        self.x_u_ph = tf.placeholder(tf.float32, [self.batch_size_U] + self.IMAGE_DIM, name='unlabeled_images')
        self.x_u_c_ph = tf.placeholder(tf.float32, [self.batch_size_U] + self.IMAGE_DIM, name='unlabeled_images_for_c')
        self.y_u_ph = tf.placeholder(tf.float32, [self.batch_size_U, self.num_class], name='unlabeled_tmp')
        
        self.embimg = tf.placeholder(tf.float32, [self.len] + self.IMAGE_DIM, name='embimg')
        self.embimg_y = tf.placeholder(tf.float32, [self.len, self.num_class], name='embimg_label')
        
        self.weights = tf.placeholder_with_default(0.0, [], name='weight')
        self.keep_prob_first = tf.placeholder(tf.float32, name='keep_prob_first')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tf.summary.scalar("Loss/recon_wieght", self.weights)
        
        self.build(is_train=is_train)
    
    def get_feed_dict_withunlabel(self, batch_chunk, batch_chunk_unlabel, z, z_linspace, y, step=None, is_training=None):
        fd = {
            self.x_l_ph: batch_chunk['image'],
            self.y_l_ph: batch_chunk['label'],
            self.z_g_ph: z,
            self.z_g_ph_linspace: z_linspace,
            self.y_g_ph: y,
            self.x_u_ph: batch_chunk_unlabel['image'][:self.batch_size_U],
            self.x_u_c_ph: batch_chunk_unlabel['image'][self.batch_size_U:self.batch_size_U + self.batch_size_U],
            self.y_u_ph: y
        }
        
        if is_training is not None:
            fd[self.is_training] = is_training
            
        if is_training:
            fd[self.keep_prob_first] = 0.2
            fd[self.keep_prob] = 0.5
        else:
            fd[self.keep_prob_first] = 1.0
            fd[self.keep_prob] = 1.0
            
        if step > 50000:
            fd[self.weights] = 1.0
        
        return fd
        
    def _entropy(self, logits):
        with tf.name_scope('Entropy'):
            probs = tf.nn.softmax(logits)
            ent = tf.reduce_mean(- tf.reduce_sum(probs * logits, axis=1, keepdims=True) \
                                 + tf.reduce_logsumexp(logits, axis=1, keepdims=True))
        return ent

    def build(self, is_train=True):

        n = self.num_class
        
        # build loss and accuracy {{{
        def build_loss(D_real, D_real_logits, D_real_logits_org, D_real_FM, D_real_tf, D_real_logits_tf, D_fake_bad, D_fake_logits_bad, D_fake_bad_FM, D_fake_good, D_fake_logits_good, D_fake_good_tf, D_fake_logits_good_tf, D_unl, D_unl_logits, D_unl_logits_noaug, D_unl_FM, D_unl_hard, D_unl_tf, D_unl_logits_tf, C_unl_hard, C_unl_tf, C_unl_logits_tf, x, fake_image_bad, fake_image_good, label, y_g_ph, x_u_ph, y_u, C):
            
            # Good GANs
            # Discriminator/classifier loss
            c_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=D_real_logits, labels=label))
                
            c_loss_fake_bad = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(D_fake_logits_bad, axis = 1)))
            c_loss_unl_bad = - 0.5 * tf.reduce_mean(tf.reduce_logsumexp(D_unl_logits, axis = 1)) + 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(D_unl_logits, axis = 1)))
            
            probs = tf.nn.softmax(D_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl_rein = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits_tf),
                                                               logits=D_unl_logits_tf), axis = 1)  ## C fools D
            c_loss_unl_rein = 0.5 * tf.reduce_mean(tf.multiply(p, c_loss_unl_rein))
            
            c_loss_fake_good_pseudo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_fake_logits_good, labels=y_g_ph))

            d_loss_real_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(D_real_logits_tf), logits = D_real_logits_tf))
            d_loss_fake_good_tf = 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_fake_logits_good_tf), logits = D_fake_logits_good_tf))

            d_wgangp_penalty = 10 * wgangp_penalty(C, x, fake_image_good, label)
            d_l2_penalty = 1e-9 * l2_penalty(C)
            
            d_unl_add=tf.constant(1e-10)
            d_loss_unl_tf_tmp=tf.constant(0.0)
            for i in range(C_unl_logits_tf.shape[0]):
                d_loss_unl_tf_tmp += tf.where(tf.equal(tf.argmax(tf.one_hot(D_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(0,dtype=tf.int64)),tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_unl_logits_tf[i]), logits = D_unl_logits_tf[i])),tf.constant(0.0))
                d_loss_unl_tf_tmp += tf.where(tf.equal(tf.argmax(tf.one_hot(D_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(3,dtype=tf.int64)),tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_unl_logits_tf[i]), logits = D_unl_logits_tf[i])),tf.constant(0.0))
                d_unl_add += tf.where(tf.equal(tf.argmax(tf.one_hot(D_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(0,dtype=tf.int64)),tf.constant(1.0),tf.constant(0.0))
                d_unl_add += tf.where(tf.equal(tf.argmax(tf.one_hot(D_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(3,dtype=tf.int64)),tf.constant(1.0),tf.constant(0.0))
            d_loss_unl_tf = 0.5 * d_loss_unl_tf_tmp / d_unl_add
            
            c_unl_add=tf.constant(1e-10)
            d_loss_unl_tf_c_tmp=tf.constant(0.0)
            for i in range(C_unl_logits_tf.shape[0]):
                d_loss_unl_tf_c_tmp += tf.where(tf.equal(tf.argmax(tf.one_hot(C_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(1,dtype=tf.int64)),tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(C_unl_logits_tf[i]), logits = C_unl_logits_tf[i])),tf.constant(0.0))
                d_loss_unl_tf_c_tmp += tf.where(tf.equal(tf.argmax(tf.one_hot(C_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(2,dtype=tf.int64)),tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(C_unl_logits_tf[i]), logits = C_unl_logits_tf[i])),tf.constant(0.0))
                d_loss_unl_tf_c_tmp += tf.where(tf.equal(tf.argmax(tf.one_hot(C_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(4,dtype=tf.int64)),tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(C_unl_logits_tf[i]), logits = C_unl_logits_tf[i])),tf.constant(0.0))
                c_unl_add += tf.where(tf.equal(tf.argmax(tf.one_hot(C_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(1,dtype=tf.int64)),tf.constant(1.0),tf.constant(0.0))
                c_unl_add += tf.where(tf.equal(tf.argmax(tf.one_hot(C_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(2,dtype=tf.int64)),tf.constant(1.0),tf.constant(0.0))
                c_unl_add += tf.where(tf.equal(tf.argmax(tf.one_hot(C_unl_hard, depth = self.num_class),axis=1)[i], tf.constant(4,dtype=tf.int64)),tf.constant(1.0),tf.constant(0.0))
            d_loss_unl_tf_c = 0.5 * d_loss_unl_tf_c_tmp / c_unl_add
            
            # Conditional entropy
            c_ent = 0.1 * tf.reduce_mean(tf.distributions.Categorical(logits=D_unl_logits).entropy())
            c_loss_balance = 0.05 * self._balance_entropy(D_unl_logits)
            
            # Batch Nuclear-norm Maximization
            c_bnm = - 0.1 * tf.reduce_sum(tf.svd(D_unl, compute_uv = False)) / self.batch_size_U

            c_loss = c_loss_real + self.weights * c_loss_fake_good_pseudo + c_loss_unl_rein
            c_loss += c_loss_fake_bad + c_loss_unl_bad + c_bnm
            
            d_loss = d_loss_real_tf + d_loss_fake_good_tf + d_loss_unl_tf_c + d_loss_unl_tf
            
            # Feature matching loss
            F_match_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(D_unl_FM,axis=0) - tf.reduce_mean(D_fake_bad_FM,axis=0)))
            
            # # entropy term via pull-away term
            feat_norm = D_fake_bad_FM / tf.norm(D_fake_bad_FM, ord='euclidean', axis=1, \
                                                 keepdims=True)
            cosine = tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]])
            mask = tf.ones(tf.shape(cosine)) - tf.diag(tf.ones(tf.shape(cosine)[0]))
            square = tf.reduce_sum(tf.square(tf.multiply(cosine, mask)))
            divident = tf.cast(tf.shape(cosine)[0] * (tf.shape(cosine)[0] - 1), tf.float32)
            G_pt = 0.1 * tf.divide(square, divident)
            
            g_loss_bad = F_match_loss + G_pt

            g_loss_good = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(D_fake_logits_good_tf), logits = D_fake_logits_good_tf))
            
            g_loss = g_loss_good + g_loss_bad
            
            GAN_loss = tf.reduce_mean(d_loss + g_loss + c_loss)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(D_real_logits_org, axis = 1),
                                          tf.argmax(label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            return c_loss_real, c_loss_fake_bad, c_loss_unl_bad, c_loss_unl_rein, c_loss_fake_good_pseudo, d_loss_real_tf, d_loss_fake_good_tf, d_loss_unl_tf, d_loss_unl_tf_c, c_ent, c_loss_balance, c_bnm, F_match_loss, G_pt, g_loss_bad, g_loss_good, c_loss, d_loss, g_loss, Vat_loss, GAN_loss, accuracy, d_loss_unl_tf_c_tmp, c_unl_add, d_loss_unl_tf_tmp, d_unl_add, d_wgangp_penalty, d_l2_penalty, Pi_Model, Pi_Model_scl
        # }}}

        # Generator {{{
        # =========
        G_bad = Generator('Generator_bad', self.h, self.w, self.c, self.is_training, use_sn = False)
        G_good = gGenerator('Generator_good', self.h, self.w, self.c, self.is_training, use_sn = False)

        self.fake_image_bad = G_bad(self.z_g_ph)
        self.fake_image_good = G_good(self.z_g_ph, self.y_g_ph)
        # }}}

        # Generator {{{
        # =========
        # output of C for real images
        C = Classifier_proD('Classifier_proD', self.num_class, use_sn = True)

        # Discriminator {{{
        # =========
        x_r = DiffAugment(self.x_l_ph)
        #x_r = self.x_l_ph
        _, _, D_real_tf, D_real_logits_tf, _  = C(x_r, self.y_l_ph, self.keep_prob_first, self.keep_prob)
        D_real, D_real_logits, _, _, D_real_FM = C(x_r, self.y_l_ph, self.keep_prob_first, self.keep_prob)
        
        self.real_predict, D_real_logits_org, _, _, _ = C(self.x_l_ph, self.y_l_ph, self.keep_prob_first, self.keep_prob)
        _, _, _, _, _ = C(self.embimg, self.embimg_y, self.keep_prob_first, self.keep_prob)
        self.real_score, _, _, _, _ = C(self.x_scorecam, self.y_scorecam, self.keep_prob_first, self.keep_prob)

        # output of D for generated examples
        x_bG = DiffAugment(self.fake_image_bad)
        #x_bG = self.fake_image_bad
        D_fake_bad, D_fake_logits_bad, _, _, D_fake_bad_FM  = C(x_bG, self.y_g_ph, self.keep_prob_first, self.keep_prob)
        
        x_gG = DiffAugment(self.fake_image_good)
        #x_gG = self.fake_image_good
        D_fake_good, D_fake_logits_good, _, _, D_fake_good_FM = C(x_gG, self.y_g_ph, self.keep_prob_first, self.keep_prob)
        _, _, D_fake_good_tf, D_fake_logits_good_tf, _ = C(x_gG, self.y_g_ph, self.keep_prob_first, self.keep_prob)
        
        # output of D for unlabeled examples (negative example)
        x_d = DiffAugment(self.x_u_ph)
        #x_d = self.x_u_ph
        D_unl, D_unl_logits, _, _, D_unl_FM = C(x_d, self.y_u_ph, self.keep_prob_first, self.keep_prob)
        D_unl_tmp, D_unl_logits_noaug, _, _, _ = C(self.x_u_ph, self.y_u_ph, self.keep_prob_first, self.keep_prob)
        D_unl_hard = tf.argmax(D_unl_tmp, axis = 1)
        _, _, D_unl_tf, D_unl_logits_tf, _ = C(x_d, tf.one_hot(D_unl_hard, depth = self.num_class), self.keep_prob_first, self.keep_prob)
        
        x_c = DiffAugment(self.x_u_c_ph)
        #x_c = self.x_u_c_ph
        C_unl, C_unl_logits, _, _, C_unl_FM = C(x_c, self.y_u_ph, self.keep_prob_first, self.keep_prob)
        C_unl_tmp, _, _, _, _ = C(self.x_u_c_ph, self.y_u_ph, self.keep_prob_first, self.keep_prob)
        C_unl_hard = tf.argmax(C_unl_tmp, axis = 1)
        _, _, C_unl_tf, C_unl_logits_tf, _ = C(x_c, tf.one_hot(C_unl_hard, depth = self.num_class), self.keep_prob_first, self.keep_prob)

        self.all_preds = D_real_logits_org
        self.all_targets = self.y_l_ph # 错误大概率就是数据和标签没对上。
        # }}}
        
        self.real_activations = D_real_logits
        self.fake_activations = D_fake_logits_good
        
        self.c_loss_real, self.c_loss_fake_bad, self.c_loss_unl_bad, self.c_loss_unl_rein, self.c_loss_fake_good_pseudo, self.d_loss_real_tf, self.d_loss_fake_good_tf, self.d_loss_unl_tf, self.d_loss_unl_tf_c, self.c_ent, self.c_loss_balance, self.c_bnm, self.F_match_loss, self.G_pt, self.g_loss_bad, self.g_loss_good, self.c_loss, self.d_loss, self.g_loss, self.Vat_loss, self.GAN_loss, self.accuracy, self.d_loss_unl_tf_c_tmp, self.c_unl_add, self.d_loss_unl_tf_tmp, self.d_unl_add, self.d_wgangp_penalty, self.d_l2_penalty, self.Pi_Model, self.Pi_Model_scl = build_loss(D_real, D_real_logits, D_real_logits_org, D_real_FM, D_real_tf, D_real_logits_tf, D_fake_bad, D_fake_logits_bad, D_fake_bad_FM, D_fake_good, D_fake_logits_good, D_fake_good_tf, D_fake_logits_good_tf, D_unl, D_unl_logits, D_unl_logits_noaug, D_unl_FM, D_unl_hard, D_unl_tf, D_unl_logits_tf, C_unl_hard, C_unl_tf, C_unl_logits_tf, self.x_l_ph, self.fake_image_bad, self.fake_image_good, self.y_l_ph, self.y_g_ph, x_d, self.y_u_ph, C)

        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
