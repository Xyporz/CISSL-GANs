from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
# Messidor Kaggle DDR IDRID2018
from six.moves import xrange
from pprint import pprint
import h5py
#import nni
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from input_ops import create_input_ops
from util import log
from config import argparser, get_params
from tensorflow.contrib.tensorboard.plugins import projector
import cv2
import time
from tensorflow.python.framework import ops
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, fbeta_score, cohen_kappa_score
from fid import compute_is_from_activations, compute_fid_from_activations, compute_kid_from_activations, compute_prd_from_embedding, inception_transform_np, plot
#from prdc import compute_prdc

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

class Trainer(object):

    def __init__(self, config, model, dataset_train, dataset_train_unlabel, dataset_test):
        self.config = config
        self.model = model
        hyper_parameter_str = '{}_lr_g_{}_d_{}_update_G{}D{}'.format(
            config["dataset"], config["learning_rate_g"], config["learning_rate_d"], 
            config["update_rate"], 1
        )
        self.train_dir = './train_dir/ICPR_128/%s-%s-%s' % (
            config["prefix"],
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )
    
        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)
        
        self.batch_size = config["batch_size_L"]
        
        # --- input ops ---
        self.batch_train = create_input_ops(
            dataset_train, config["batch_size_L"])

        self.batch_train_unlabel = create_input_ops(
            dataset_train_unlabel, config["batch_size_U"] * 2)

        self.batch_test = create_input_ops(
            dataset_test, config["batch_size_L"])
    
        length_pooledfeature = 512
        length_disfeature = 512 #45056
        self.embedding = tf.Variable(tf.zeros((config["img"].shape[0], length_disfeature)), trainable=False, name='embedding')
        self.mid_vari = tf.placeholder(tf.float32, [config["img"].shape[0], length_disfeature],name="mid_vari")
        self.mid_op = tf.assign(self.embedding, self.mid_vari)
        self.tembimg = config["img"]
        
        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()

        gG_var = [v for v in all_var if v.name.startswith(('Generator_g'))]
        log.warning("********* gG_var ********** ")
        slim.model_analyzer.analyze_vars(gG_var, print_info=True)
        
        bG_var = [v for v in all_var if v.name.startswith(('Generator_b'))]
        log.warning("********* bG_var ********** ")
        slim.model_analyzer.analyze_vars(bG_var, print_info=True)
        
        c_var = [v for v in all_var if v.name.startswith(('Classifier'))]
        log.warning("********* c_var ********** ")
        slim.model_analyzer.analyze_vars(c_var, print_info=True)
        
        rem_var = (set(all_var) - set(gG_var) - set(bG_var) - set(c_var))
        log.error([v.name for v in rem_var])
        assert not rem_var

        self.gG_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.g_loss_good,
            global_step=self.global_step,
            learning_rate=config["learning_rate_g"],
            optimizer=tf.train.AdamOptimizer(beta1=0.5,beta2=0.999),
            clip_gradients=20.0,
            name='gG_optimize_loss',
            variables=gG_var
        )
        
        self.bG_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.g_loss_bad,
            global_step=self.global_step,
            learning_rate=config["learning_rate_g"],
            optimizer=tf.train.AdamOptimizer(beta1=0.5,beta2=0.999),
            clip_gradients=20.0,
            name='bG_optimize_loss',
            variables=bG_var
        )
        
        self.c_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.c_loss + self.model.d_loss,
            global_step=self.global_step,
            learning_rate=config["learning_rate_d"],
            optimizer=tf.train.AdamOptimizer(beta1=0.5,beta2=0.999),
            clip_gradients=20.0,
            name='c_optimize_loss',
            variables=c_var
        )
        
        self.c_optimizer_only = tf.contrib.layers.optimize_loss(
            loss=self.model.c_loss,
            global_step=self.global_step,
            learning_rate=config["learning_rate_d"],
            optimizer=tf.train.AdamOptimizer(beta1=0.5,beta2=0.999),
            clip_gradients=20.0,
            name='c_optimize_loss_only',
            variables=c_var
        )
        
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        
        pconfig = projector.ProjectorConfig()
        embed = pconfig.embeddings.add()
        embed.tensor_name = self.embedding.name
        embed.metadata_path = '/home/ubuntu/xieyingpeng/GBGANs/GBGANs_ICPR_N/datasets/metadata.tsv'
        projector.visualize_embeddings(self.summary_writer, pconfig)
        
        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.99),
            device_count={'GPU': 1}
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)
        
        self.ckpt_path = config["checkpoint"]
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")
            
    def train(self):
        log.infov("Training Starts!")
        log.infov(self.batch_train)
        log.infov(self.batch_train_unlabel)
        step = self.session.run(self.global_step)#从0开始，global_step是一个Variable类型的参数，在所有的网络参数结束梯度更新后，global_step会自增加一，第一次更新置0。
        
        for s in xrange(self.config["max_training_steps"]):
            step, accuracy, d_loss, g_loss, step_time, prediction_train, gt_train, Image_Real = \
                self.run_single_step(self.batch_train, self.batch_train_unlabel, step=s, is_train = True)
                
            if s % self.config["log_step"] == self.config["log_step"] - 1:
                self.log_step_message(step + 1, accuracy, d_loss, g_loss, step_time, is_train=True)
                
            # periodic inference
            if s % self.config["test_sample_step"] == self.config["test_sample_step"] - 1:
            
                accuracy_mean, recall, precision, f1, kappa = [], [], [], [], []
                
                #for _ in range(1):
                accuracy, summary, d_loss, g_loss, step_time, prediction_test, gt_test, Image_Generate_test = \
                    self.run_test(self.batch_test, self.batch_train_unlabel, is_train=False, step=s)
                self.log_step_message(step + 1, accuracy, d_loss, g_loss,
                                      step_time, is_train=False)

                self.summary_writer.add_summary(summary, global_step=step + 1)
                
                y_true = np.argmax(prediction_test, axis=1)
                y_pred = np.argmax(gt_test, axis=1)
                
                accuracy_mean.append(accuracy_score(y_true, y_pred))
                recall.append(recall_score(y_true, y_pred, average="macro"))
                precision.append(precision_score(y_true, y_pred, average="macro"))
                f1.append(f1_score(y_true, y_pred, average="macro"))
                kappa.append(cohen_kappa_score(y_true, y_pred))
                

            if s % self.config["output_save_step"] == self.config["output_save_step"] - 1 and s != self.config["max_training_steps"]-1:
                log.infov("Saved checkpoint at %d", step + 1)
                self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step + 1)
            
            if s == self.config["max_training_steps"] - 1:
                emb = self.session.run(self.model.tsne, feed_dict = {self.model.embimg: self.tembimg, self.model.keep_prob_first: 1.0, self.model.keep_prob: 1.0})
                self.session.run(self.mid_op,feed_dict={self.mid_vari: emb})
                log.infov("Saved checkpoint at %d", step + 1)
                self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step + 1)
                
                #nni.report_final_result(np.mean(np.sort(metric)[-10:]))
            
    def run_single_step(self, batch, batch_unlabel, step=None, is_train=True):
    
        _start_time = time.time()

        batch_chunk = self.session.run(batch)
        batch_chunk_unlabel = self.session.run(batch_unlabel)
        
        z = np.random.uniform(low = -1.0, high = 1.0, size=(self.config["batch_size_G"], self.config["n_z"])).astype(np.float32)
        z_tmp = np.random.uniform(low = -1.0, high = 1.0, size=(self.config["n_z"])).astype(np.float32)
        z_linspace = np.linspace(z[0], z_tmp, 10)
        #z = np.random.normal(loc=0.0, scale=1.0, size=(self.config["batch_size_G"], self.config["n_z"])).astype(np.float32)
        y_temp = np.random.randint(low = 0, high = self.config["num_class"], size = (self.config["batch_size_G"]))
        y = np.zeros((self.config["batch_size_G"], self.config["num_class"]))
        y[np.arange(self.config["batch_size_G"]), y_temp] = 1
        
        fetch = [self.global_step, self.model.accuracy,
                 self.model.d_loss, self.model.g_loss,
                 self.model.x_l_ph, self.model.output_conv, self.model.grads_val, self.model.guide_grad, self.model.grad_val_plusplus, self.model.cls, self.model.logit_cam,
                 self.model.all_preds, self.model.all_targets]
        
        if step % (self.config["update_rate"]+1) == 0:
        # Train the generator
            fetch.append(self.c_optimizer)
        elif step % (self.config["update_rate"]+1) == 1:
        # Train the discriminator
            fetch.append(self.gG_optimizer)
        elif step % (self.config["update_rate"]+1) == 2:
            fetch.append(self.bG_optimizer)

        fetch_values = self.session.run(fetch,
            feed_dict = self.model.get_feed_dict_withunlabel(batch_chunk, batch_chunk_unlabel, z, z_linspace, y, step=step, is_training = is_train))
        # log.error(fetch_values[8]) #该值是因为上面的append生成器判别器产生的。
        [step, accuracy, d_loss, g_loss, Image_Real, output_conv, grads_val, guide_grad, grad_val_plusplus, cls, logit_cam,\
         all_preds, all_targets] = fetch_values[:13]
        
        _end_time = time.time()
        
        return step, accuracy, d_loss, g_loss, \
            (_end_time - _start_time), all_preds, all_targets, Image_Real
            

    def run_test(self, batch, batch_unlabel, is_train=False, step=None):
    
        _start_time = time.time()
        
        batch_chunk = self.session.run(batch)
        batch_chunk_unlabel = self.session.run(batch_unlabel)
        
        z = np.random.uniform(low = -1.0, high = 1.0, size=(self.config["batch_size_G"], self.config["n_z"])).astype(np.float32)
        z_tmp = np.random.uniform(low = -1.0, high = 1.0, size=(self.config["n_z"])).astype(np.float32)
        z_linspace = np.linspace(z[0], z_tmp, 10)
        y_temp = np.random.randint(low = 0, high = self.config["num_class"], size = (self.config["batch_size_G"]))
        y = np.zeros((self.config["batch_size_G"], self.config["num_class"]))
        y[np.arange(self.config["batch_size_G"]), y_temp] = 1
        
        [accuracy, summary, d_loss, g_loss, all_preds, all_targets, Image_Generate] = self.session.run(
            [self.model.accuracy, self.summary_op, self.model.d_loss,
             self.model.g_loss, self.model.all_preds, self.model.all_targets, self.model.Image_Generate],
            feed_dict=self.model.get_feed_dict_withunlabel(batch_chunk, batch_chunk_unlabel, z, z_linspace, y, step=step, is_training = is_train))
        
        _end_time = time.time()

        return accuracy, summary, d_loss, g_loss, (_end_time - _start_time), all_preds, all_targets, Image_Generate

    def log_step_message(self, step, accuracy, d_loss, g_loss,
                         step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "D loss: {d_loss:.5f} " +
                "G loss: {g_loss:.5f} " +
                "Accuracy: {accuracy:.5f} "
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step = step,
                         d_loss = d_loss,
                         g_loss = g_loss,
                         accuracy = accuracy,
                         sec_per_batch = step_time,
                         instance_per_sec = self.batch_size / step_time)
               )

def main():
    try:
        #tuner_params = nni.get_next_parameter()
        #params = vars(get_params())
        #params.update(tuner_params)
        
        params, args = get_params()
        params = vars(params)
        
        config, model, dataset_train, dataset_train_unlabel, dataset_val, dataset_test = argparser(params, is_train=True)
        trainer = Trainer(config, model, dataset_train, dataset_train_unlabel, dataset_val)

        log.info("dataset: %s, learning_rate_g: %f, learning_rate_d: %f",
                    config["dataset"], config["learning_rate_g"], config["learning_rate_d"])
                    
        trainer.train()
        
    except Exception as exception:
        log.exception(exception)
        raise

if __name__ == '__main__':
    
    main()
    
