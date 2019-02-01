#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:22:39 2019

@author: jim
"""

import tensorflow as tf

class SongModel(object):
    def __init__(self, setting):
        self.setting = setting
        
    def add_placeholder(self):
        self.song_input = tf.placeholder(dtype=tf.int32, 
                                         shape=[None, self.setting.num_song], 
                                         name='Song_Input')
        
    # Song embedding layer
    def create_embedding(self):
        with tf.variable_scope('SONG_EMBEDDING'):
            initial_value = tf.truncated_normal([self.setting.song_size, 
                                                 self.setting.song_embedding_dim], 
                                                mean=0.0, 
                                                stddev=1.0)
            self.embedding = tf.Variable(initial_value,
                                         trainable=True, 
                                         name='song_embedding_weight')
            print('song_embedding')
            print(self.embedding.shape)
            self.embedding_output = tf.nn.embedding_lookup(params=self.embedding,
                                                           ids=self.song_input)
            print('song_embedding_output')
            print(self.embedding_output.shape)
            
    
    # Define Basic RNN Cell
    def basic_rnn_cell(self, rnn_size):
        cell = tf.nn.rnn_cell.GRUCell(rnn_size)
        return cell
    # Bi-GRU
    def create_bi_GRU(self):
        with tf.variable_scope('SONG_Bi_GRU'):
            # Define Forward RNN Cell
            with tf.name_scope('fw_rnn'):
                fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([self.basic_rnn_cell(self.setting.song_rnn_size) 
                                                            for _ in range(self.setting.song_num_rnn_layers)])
                if self.setting.song_keep_probability is not None:
                    fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell, 
                                                                input_keep_prob=self.setting.song_keep_probability)
            # Define Backward RNN Cell
            with tf.name_scope('bw_rnn'):
                bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([self.basic_rnn_cell(self.setting.song_rnn_size) 
                                                            for _ in range(self.setting.song_num_rnn_layers)])
                if self.setting.song_keep_probability is not None:
                    bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell, 
                                                                input_keep_prob=self.setting.song_keep_probability)
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
            print('song_bi_gru_output')
            print(self.bi_gru_output.shape)
    
    def make_attention_cell(self):
        #Attention Layer
        with tf.name_scope('SONG_ATTENTION'):
            input_shape = self.bi_gru_output.shape
            sequence_size = input_shape[1].value
            hidden_size = input_shape[2].value
            
            initial_value = tf.truncated_normal([hidden_size, sequence_size], 
                                                mean=0.0, 
                                                stddev=0.1)
            self.attention_w = tf.Variable(initial_value,
                                           trainable=True, 
                                           name='song_attention_weight')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(self.bi_gru_output[:, t, :], self.attention_w))
                z_list.append(u_t)
            # Transform to batch_size * sequence_size
            self.song_attention_output = tf.concat(z_list, axis=1)
            print('song_attention_output')
            print(self.song_attention_output.shape)
                 