#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 02:51:03 2019

@author: jim
"""

import tensorflow as tf
import Setting
setting = Setting.setting()

class TagModel(object):
    def add_placeholder(self):
        self.tag_input = tf.placeholder(dtype=tf.int32, 
                                         shape=[None, setting.num_song, setting.num_tag], 
                                         name='Tag_Input')
        
    # Tag one hot layer
    def add_one_hot_layer(self):
        with tf.variable_scope('TAG_ONE_HOT'):
            onehot_input = tf.one_hot(indices=self.tag_input, 
                                      depth=setting.tag_size,
                                      name='tag_one_hot')
            print('onehot_input')
            print(onehot_input.shape)
            
    # Tag embedding layer
    def create_embedding(self):
        with tf.variable_scope('TAG_EMBEDDING'):
            initial_value = tf.truncated_normal([setting.tag_size, 
                                                 setting.tag_embedding_dim], 
                                                mean=0.0, 
                                                stddev=1.0)
            self.embedding = tf.Variable(initial_value,
                                         trainable=True, 
                                         name='tag_embedding_weight')
            print('tag_embedding')
            print(self.embedding.shape)
            embedding_tag = []
            #song_idx tag_idx
            for s, t in enumerate([3,2,1,1,2]):
                avg_tag = tf.reduce_mean(tf.nn.embedding_lookup(params=self.embedding,
                                                                ids=self.tag_input[:,s:s+1,0:t]), 2)
                #print('avg_tag')
                #print(avg_tag.shape)
                embedding_tag.append(avg_tag)
            self.embedding_output = tf.concat(embedding_tag, axis=1)
            print('tag_embedding_output')
            print(self.embedding_output.shape)
            
    # Define Basic RNN Cell
    def basic_rnn_cell(self, rnn_size):
        cell = tf.nn.rnn_cell.GRUCell(rnn_size)
        return cell
    # Bi-GRU
    def create_bi_GRU(self):
        with tf.variable_scope('TAG_Bi_GRU'):
            # Define Forward RNN Cell
            with tf.name_scope('fw_rnn'):
                fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([self.basic_rnn_cell(setting.tag_rnn_size) 
                                                            for _ in range(setting.tag_num_rnn_layers)])
                if setting.tag_keep_probability is not None:
                    fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell, 
                                                                input_keep_prob=setting.tag_keep_probability)
            # Define Backward RNN Cell
            with tf.name_scope('bw_rnn'):
                bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([self.basic_rnn_cell(setting.tag_rnn_size) 
                                                            for _ in range(setting.tag_num_rnn_layers)])
                if setting.tag_keep_probability is not None:
                    bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell, 
                                                                input_keep_prob=setting.tag_keep_probability)
            with tf.name_scope('bi_gru'):
                rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=fw_rnn_cell,
                        cell_bw=bw_rnn_cell,
                        inputs=self.embedding_output,
                        #sequence_length=self.rnn_seq_len,
                        dtype=tf.float32)
                # In case of Bi-RNN, concatenate the forward and the backward RNN outputs
                if isinstance(rnn_output, tuple):
                    self.bi_gru_output = tf.concat(rnn_output, 2)
            print('tag_bi_gru_output')
            print(self.bi_gru_output.shape)
            
    def make_attention_cell(self):
        #Attention Layer
        with tf.name_scope('TAG_ATTENTION'):
            input_shape = self.bi_gru_output.shape
            sequence_size = input_shape[1].value
            hidden_size = input_shape[2].value
            
            initial_value = tf.truncated_normal([hidden_size, sequence_size], 
                                                mean=0.0, 
                                                stddev=0.1)
            self.attention_w = tf.Variable(initial_value,
                                           trainable=True, 
                                           name='tag_attention_weight')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(self.bi_gru_output[:, t, :], self.attention_w))
                z_list.append(u_t)
            # Transform to batch_size * sequence_size
            self.tag_attention_output = tf.concat(z_list, axis=1)
            print('tag_attention_output')
            print(self.tag_attention_output.shape)
