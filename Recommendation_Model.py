#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:22:39 2019

@author: jim
"""

import tensorflow as tf
import numpy as np
import Setting
from SongModel import SongModel
from TagModel import TagModel
import tqdm

setting = Setting.setting()
song_model = SongModel()
tag_model = TagModel()

tf.reset_default_graph()

class Recommendation_Model(Setting.setting):
    def __init__(self):
        tf.reset_default_graph()
        
    def add_placeholder(self):
        self.label_idx = tf.placeholder(dtype=tf.int32,
                                        shape=[None, 1],
                                        name='label_idx')
        
    def create_fully_layer(self):
        # Fully connected layer setting
        self.FC_input = tf.concat(axis=1, values=[song_model.song_attention_output, 
                                                  tag_model.tag_attention_output])
        print('FC_input')
        print(self.FC_input.shape)
        with tf.name_scope('FULLY_CONNECTED_LAYER'):
            initial_value = tf.truncated_normal([self.FC_input.shape[1].value, 
                                                 setting.fully_neuron],
                                                stddev=0.1)
            self.fully_weight = tf.Variable(initial_value, name='fully_weight')
            self.fully_bais = tf.Variable(tf.zeros(shape=[setting.fully_neuron]), name='fully_bais')
            self.FC_output = tf.nn.relu(tf.matmul(self.FC_input, self.fully_weight) + self.fully_bais)
            print('FC_output')
            print(self.FC_output.shape)
        # Output layer setting
        with tf.name_scope('OUTPUT_LAYER'):
            initial_value = tf.truncated_normal([self.FC_output.shape[1].value, setting.output_dim],
                                                stddev=0.1)
            self.output_weight = tf.Variable(initial_value, name='output_weight')
            self.output_bais = tf.Variable(tf.zeros(shape=[setting.output_dim]), name='output_bais')
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
            self.onehot_label = tf.one_hot(self.label_idx, setting.song_size, name='label_one_hot')
            self.onehot_label = tf.reshape(self.onehot_label, [-1, setting.song_size])
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
            optimizer = tf.train.AdamOptimizer(learning_rate=setting.learning_rate)
            # Gradients
            if setting.grad_clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, setting.grad_clip)
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
        song_model.add_placeholder()
        #song_model.add_one_hot_layer()
        song_model.create_embedding()
        song_model.create_bi_GRU()
        song_model.make_attention_cell()
        
        #Tag Model
        tag_model.add_placeholder()
        #tag_model.add_one_hot_layer()
        tag_model.create_embedding()
        tag_model.create_bi_GRU()
        tag_model.make_attention_cell()
        
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
        self.saver.save(self.sess, setting.save_path + model_name + '.ckpt')

    def train(self, Song_X, Tag_X, Song_Y, Vali_Song_X, Vali_Tag_X, Vali_Song_Y):
        assert len(Song_X) == len(Song_Y)
        for epoch in range(setting.epochs):
            print('------------------ Epoch {} of {} ------------------'.format(epoch + 1,
                  setting.epochs))
            # shuffle the input data before every epoch.
            shuffle_indices = np.random.permutation(len(Song_X))
            Song_X = Song_X[shuffle_indices]
            Song_Y = Song_Y[shuffle_indices]
            #batch
            Train_sum = 0
            for i in tqdm.tqdm(range(Song_X.shape[0]//setting.batch_size + 1)):
                start = i * setting.batch_size
                end = (i + 1) * setting.batch_size
                if end > Song_X.shape[0]:
                    end = Song_X.shape[0]
                if start == Song_X.shape[0]:
                    continue
                #song_input label_idx rnn_seq_len
                Song_input = Song_X[start:end,:]
                Tag_input = Tag_X[start:end,:,:]
                Song_label = Song_Y[start:end,:]
                self.feed_dict = {song_model.song_input: Song_input, 
                                  tag_model.tag_input: Tag_input, 
                                  self.label_idx: Song_label
                                 }
                _, train_loss, train_accuracy, output = self.sess.run([self.train_op, 
                                                               self.loss, 
                                                               self.accuracy,
                                                               self.Output],
                                                              feed_dict=self.feed_dict)
                Train_sum = Train_sum + train_accuracy
            print("train loss: {:.8f} train accuracy: {}\n".format(train_loss, 
                  Train_sum/(Song_X.shape[0]//setting.batch_size)))
            
            #Validation
            Vali_sum = 0
            for i in range(Vali_Song_X.shape[0]//setting.batch_size + 1):
                start = i * setting.batch_size
                end = (i + 1) * setting.batch_size
                if end > Vali_Song_X.shape[0]:
                    end = Vali_Song_X.shape[0]
                if start == Vali_Song_X.shape[0]:
                    continue
                #song_input label_idx rnn_seq_len
                Vali_Song_input = Song_X[start:end,:]
                Vali_Tag_input = Tag_X[start:end,:,:]
                Vali_Song_label = Song_Y[start:end,:]
                vali_feed = {song_model.song_input: Vali_Song_input, 
                             tag_model.tag_input: Vali_Tag_input, 
                             self.label_idx: Vali_Song_label
                             }
                vali_acc = self.sess.run(self.accuracy, feed_dict=vali_feed)
                Vali_sum = Vali_sum + vali_acc
            print("vali accuracy: ", Vali_sum / (Vali_Song_X.shape[0]//setting.batch_size))
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        