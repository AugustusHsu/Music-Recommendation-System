#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:22:39 2019

@author: jim
"""

import tensorflow as tf
import numpy as np
import tqdm

class Recommendation_Model:
    def __init__(self,
                 save_path,
                 mode='TRAIN',
                 song_size=130009,
                 num_song=5,
                 batch_size=64,
                 embedding_dim=50,
                 rnn_size=1,
                 num_rnn_layers=1,
                 fully_neuron=50,
                 output_dim=130009,
                 learning_rate=0.00006,
                 grad_clip=0.0,
                 keep_probability=None,
                 epochs=100
                 ):
        """
        Args:
            save_path:  String. 
                        path to save the tf model to in the end.
            mode:       String. 
                        'TRAIN' or 'INFER'. depending on which mode we use a different graph is created.
            num_layers: number of encoder layers. defaults to 1.
            embedding_dim: dimension of the embedding vectors in the embedding matrix.
                           every word has a embedding_dim 'long' vector.
                           defaults to 50.
            rnn_size: Integer. number of hidden units in encoder. defaults to 256.
            num_rnn_layers: Integer. number of rnn layer. defaults to 1
            learning_rate: Float.
            
            learning_rate_decay: only if exponential learning rate is used.
            learning_rate_decay_steps: Integer.
            max_lr: only used if cyclic learning rate is used.
            keep_probability: Float. defalt to None
        """
        self.song_size = song_size
        self.num_song = num_song
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.rnn_size=rnn_size
        self.num_rnn_layers=num_rnn_layers
        self.fully_neuron=fully_neuron
        self.output_dim=output_dim
        self.learning_rate=learning_rate
        self.grad_clip=grad_clip
        self.keep_probability=keep_probability
        self.epochs=epochs
    
    def add_placeholder(self):
        self.song_input = tf.placeholder(dtype=tf.int32, 
                                         shape=[None, self.num_song], 
                                         name='Song_Input')
        self.label_idx = tf.placeholder(dtype=tf.int32,
                                        shape=[None, 1],
                                        name='label_idx')
        self.rnn_seq_len = tf.placeholder(tf.int32, shape=[None], 
                                          name='seq_len')
        
    # Song one hot layer
    def add_one_hot_layer(self):
        with tf.variable_scope('SONG_ONE_HOT'):
            onehot_input = tf.one_hot(indices=self.song_input, 
                                      depth=self.song_size,
                                      name='song_one_hot')
            print('onehot_input')
            print(onehot_input.shape)
            
    # Song embedding layer
    def create_embedding(self):
        with tf.variable_scope('SONG_EMBEDDING'):
            initial_value = tf.truncated_normal([self.song_size, self.embedding_dim], 
                                                mean=0.0, 
                                                stddev=1.0)
            self.embedding = tf.Variable(initial_value,
                                         trainable=True, 
                                         name='song_embedding_weight')
            print('embedding')
            print(self.embedding.shape)
            self.embedding_output = tf.nn.embedding_lookup(params=self.embedding,
                                                           ids=self.song_input)
            print('embedding_output')
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
                fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([self.basic_rnn_cell(self.rnn_size) 
                                                            for _ in range(self.num_rnn_layers)])
                if self.keep_probability is not None:
                    fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell, 
                                                                input_keep_prob=self.keep_probability)
            # Define Backward RNN Cell
            with tf.name_scope('bw_rnn'):
                bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([self.basic_rnn_cell(self.rnn_size) 
                                                            for _ in range(self.num_rnn_layers)])
                if self.keep_probability is not None:
                    bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell, 
                                                                input_keep_prob=self.keep_probability)
            with tf.name_scope('bi_gru'):
                rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=fw_rnn_cell,
                        cell_bw=bw_rnn_cell,
                        inputs=self.embedding_output,
                        sequence_length=self.rnn_seq_len,
                        dtype=tf.float32)
                # In case of Bi-RNN, concatenate the forward and the backward RNN outputs
                if isinstance(rnn_output, tuple):
                    self.bi_gru_output = tf.concat(rnn_output, 2)
            print('bi_gru_output')
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
            self.attention_output = tf.concat(z_list, axis=1)
            print('attention_output')
            print(self.attention_output.shape)
            
    def create_fully_layer(self):
        # Fully connected layer setting
        with tf.name_scope('FULLY_CONNECTED_LAYER'):
            initial_value = tf.truncated_normal([self.attention_output.shape[1].value, self.fully_neuron],
                                                stddev=0.1)
            self.fully_weight = tf.Variable(initial_value, name='fully_weight')
            self.fully_bais = tf.Variable(tf.zeros(shape=[self.fully_neuron]), name='fully_bais')
            self.FC_output = tf.nn.relu(tf.matmul(self.attention_output, self.fully_weight) + self.fully_bais)
            print('FC_output')
            print(self.FC_output.shape)
        # Output layer setting
        with tf.name_scope('OUTPUT_LAYER'):
            initial_value = tf.truncated_normal([self.FC_output.shape[1].value, self.output_dim],
                                                stddev=0.1)
            self.output_weight = tf.Variable(initial_value, name='output_weight')
            self.output_bais = tf.Variable(tf.zeros(shape=[self.output_dim]), name='output_bais')
            self.Output = tf.matmul(self.FC_output, self.output_weight) + self.output_bais
            print('Output')
            print(self.Output.shape)
            
    # Initialize all variables
    def initialize_session(self, Model):
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session(graph=Model)
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(init_op)
    
    # Calculate cross-entropy loss
    def compute_loss(self):
        # Label one hot layer
        with tf.variable_scope('LABEL_ONE_HOT'):
            self.onehot_label = tf.one_hot(self.label_idx, self.song_size, name='label_one_hot')
            self.onehot_label = tf.reshape(self.onehot_label, [-1, self.song_size])
            print('onehot_label')
            print(self.onehot_label.shape)
        
        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            log_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.Output, labels=self.onehot_label)
            self.loss = tf.reduce_mean(log_loss)
            
    def accuracy(self):
        # Calculate accuracy
        with tf.name_scope('accuracy'):
            predictions = tf.argmax(self.Output, 1, name='predictions')
            correct_pred = tf.equal(predictions, tf.argmax(self.onehot_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    def All_Model(self):
        self.add_one_hot_layer()
        self.create_embedding()
        self.create_bi_GRU()
        self.make_attention_cell()
        self.create_fully_layer()
        self.compute_loss()
        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # Gradients
            if self.grad_clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def build_graph(self):
        Model = tf.Graph()
        #tf.reset_default_graph()
        with Model.as_default():            
            tf_config = tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=False)
            #tf_config.per_process_gpu_memory_fraction = 0.8
            #tf_config.gpu_options.allow_growth=True
            self.add_placeholder()
            self.All_Model()
            self.accuracy()
            self.initialize_session(Model)
            print('Graph built.')
    

    def train(self, Song_X, Song_Y, Vali_Song_X, Vali_Song_Y):
        assert len(Song_X) == len(Song_Y)
        for epoch in range(self.epochs + 1):
            print('-------------------- Epoch {} of {} --------------------'.format(epoch,
                                                                                    self.epochs))
            # shuffle the input data before every epoch.
            shuffle_indices = np.random.permutation(len(Song_X))
            Song_X = Song_X[shuffle_indices]
            Song_Y = Song_Y[shuffle_indices]
            #random
            random_size = 1
            for i in tqdm.tqdm(range(Song_X.shape[0]//random_size)):
                if random_size == 1:
                    start = i
                    end = i + 1
                else:
                    idx = np.random.randint(Song_X.shape[0]) - 1
                    start = min(idx, Song_X.shape[0] - 1)
                    end = min(idx + 1, Song_X.shape[0])
                #song_input label_idx rnn_seq_len
                Song_input = Song_X[start:end,:]
                #Tag_input = Tag_X[start:end,:,:]
                Song_label = Song_Y[start:end,:]
                feed_dict = {self.song_input: Song_input, 
                             #model.Tag_Part.input_x: Tag_input, 
                             self.label_idx: Song_label, 
                             self.rnn_seq_len: [5], 
                             #model.Tag_Part.keep_prob: 0.9
                            }
                _, train_loss, train_accuracy = self.sess.run([self.train_op, 
                                                               self.loss, 
                                                               self.accuracy],
                                                              feed_dict=feed_dict)
            print("train loss: {:.20f} train accuracy: {:.3f}\n".format(train_loss, train_accuracy))
            
            #Validation
            vali_feed = {self.song_input: Vali_Song_X, 
                         #model.Tag_Part.input_x: Tag_input, 
                         self.label_idx: Vali_Song_Y, 
                         self.rnn_seq_len: [5], 
                         #model.Tag_Part.keep_prob: 0.9
                         }
            vali_accuracy = self.sess.run(self.accuracy, feed_dict=vali_feed)
            print("vali accuracy: {:.3f}\n".format(vali_accuracy))
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        