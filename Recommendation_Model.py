#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:22:39 2019

@author: jim
"""

import tensorflow as tf
import numpy as np
import time
from SongModel import SongModel
from TagModel import TagModel
import tqdm

class Recommendation_Model():
    def __init__(self, setting):
        tf.reset_default_graph()
        self.setting = setting
        self.song_model = SongModel(setting)
        self.tag_model = TagModel(setting)
        
    def add_placeholder(self):
        self.label_idx = tf.placeholder(dtype=tf.int32,
                                        shape=[None, 1],
                                        name='label_idx')
        
    def create_fully_layer(self):
        # Fully connected layer setting
        self.FC_input = tf.concat(axis=1, values=[self.song_model.song_attention_output, 
                                                  self.tag_model.tag_attention_output])
        print('FC_input')
        print(self.FC_input.shape)
        with tf.name_scope('FULLY_CONNECTED_LAYER'):
            initial_value = tf.truncated_normal([self.FC_input.shape[1].value, 
                                                 self.setting.fully_neuron],
                                                stddev=0.1)
            self.fully_weight = tf.Variable(initial_value, name='fully_weight')
            self.fully_bais = tf.Variable(tf.zeros(shape=[self.setting.fully_neuron]), name='fully_bais')
            self.FC_output = tf.nn.relu(tf.matmul(self.FC_input, self.fully_weight) + self.fully_bais)
            print('FC_output')
            print(self.FC_output.shape)
        # Output layer setting
        with tf.name_scope('OUTPUT_LAYER'):
            initial_value = tf.truncated_normal([self.FC_output.shape[1].value, self.setting.output_dim],
                                                stddev=0.1)
            self.output_weight = tf.Variable(initial_value, name='output_weight')
            self.output_bais = tf.Variable(tf.zeros(shape=[self.setting.output_dim]), name='output_bais')
            self.Output = tf.matmul(self.FC_output, self.output_weight) + self.output_bais
            #self.Output = tf.nn.softmax(self.Output)
            print('Output')
            print(self.Output.shape)
            
    # Initialize all variables
    def initialize_session(self, Model):
        self.sess = tf.Session(graph=Model)
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
    
    # Calculate cross-entropy loss
    def compute_loss(self):
        # Label one hot layer
        with tf.variable_scope('LABEL_ONE_HOT'):
            self.onehot_label = tf.one_hot(self.label_idx, self.setting.song_size, name='label_one_hot')
            self.onehot_label = tf.reshape(self.onehot_label, [-1, self.setting.song_size])
            print('onehot_label')
            print(self.onehot_label.shape)
        
        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            log_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Output, 
                                                                  labels=self.onehot_label)
            self.loss = tf.reduce_mean(log_loss)
            
    def create_optimizer(self):
        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.setting.learning_rate)
            # Gradients
            if self.setting.grad_clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.setting.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)
                
    def acc(self):
        # Calculate accuracy
        with tf.name_scope('accuracy'):
            #self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.onehot_label, 1), 
            #                                    predictions=tf.argmax(self.Output, 1))
            predictions = tf.argmax(self.Output, 1, name='predictions')
            correct_pred = tf.equal(predictions, tf.argmax(self.onehot_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
    def create_saver(self):
        self.saver = tf.train.Saver()
        
    def All_Model(self):
        #Song Model
        self.song_model.add_placeholder()
        #song_model.add_one_hot_layer()
        self.song_model.create_embedding()
        self.song_model.create_bi_GRU()
        self.song_model.make_attention_cell()
        
        #Tag Model
        self.tag_model.add_placeholder()
        #tag_model.add_one_hot_layer()
        self.tag_model.create_embedding()
        self.tag_model.create_bi_GRU()
        self.tag_model.make_attention_cell()
        
        self.add_placeholder()
        self.create_fully_layer()
        self.compute_loss()
        self.create_optimizer()

    def build_graph(self):
        Model = tf.Graph()
        with Model.as_default():
            self.All_Model()
            self.create_saver()
            self.acc()
            self.initialize_session(Model)
            print('Graph built.')
            
    def save_model(self, model_name = 'Music_Recommendation'):
        self.saver.save(self.sess, self.setting.save_path + model_name + '.ckpt')

    def train(self, Song_X, Tag_X, Song_Y, Vali_Song_X, Vali_Tag_X, Vali_Song_Y):
        assert len(Song_X) == len(Song_Y)
        for epoch in range(self.setting.epochs):
            print('------------------ Epoch {} of {} ------------------'.format(epoch + 1,
                  self.setting.epochs))
            time.sleep(0.5)
            # shuffle the input data before every epoch.
            shuffle_indices = np.random.permutation(len(Song_X))
            Song_X = Song_X[shuffle_indices]
            Song_Y = Song_Y[shuffle_indices]
            #batch
            Train_sum = 0
            for i in tqdm.tqdm(range(Song_X.shape[0]//self.setting.batch_size + 1)):
                start = i * self.setting.batch_size
                end = (i + 1) * self.setting.batch_size
                if end > Song_X.shape[0]:
                    end = Song_X.shape[0]
                if start == Song_X.shape[0]:
                    continue
                #song_input label_idx rnn_seq_len
                Song_input = Song_X[start:end,:]
                Tag_input = Tag_X[start:end,:,:]
                Song_label = Song_Y[start:end,:]
                self.feed_dict = {self.song_model.song_input: Song_input, 
                                  self.tag_model.tag_input: Tag_input, 
                                  self.label_idx: Song_label
                                 }
                _, train_loss, train_accuracy, output = self.sess.run([self.train_op, 
                                                               self.loss, 
                                                               self.accuracy,
                                                               self.Output],
                                                              feed_dict=self.feed_dict)
                Train_sum = Train_sum + train_accuracy
            print("train loss: {:.8f} train accuracy: {}\n".format(train_loss, 
                  Train_sum/(Song_X.shape[0]//self.setting.batch_size)))
            
            #Validation
            Vali_sum = 0
            for i in range(Vali_Song_X.shape[0]//self.setting.batch_size + 1):
                start = i * self.setting.batch_size
                end = (i + 1) * self.setting.batch_size
                if end > Vali_Song_X.shape[0]:
                    end = Vali_Song_X.shape[0]
                if start == Vali_Song_X.shape[0]:
                    continue
                #song_input label_idx rnn_seq_len
                Vali_Song_input = Song_X[start:end,:]
                Vali_Tag_input = Tag_X[start:end,:,:]
                Vali_Song_label = Song_Y[start:end,:]
                vali_feed = {self.song_model.song_input: Vali_Song_input, 
                             self.tag_model.tag_input: Vali_Tag_input, 
                             self.label_idx: Vali_Song_label
                             }
                vali_acc = self.sess.run(self.accuracy, feed_dict=vali_feed)
                Vali_sum = Vali_sum + vali_acc
            print("vali accuracy: ", Vali_sum / (Vali_Song_X.shape[0]//self.setting.batch_size))
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        