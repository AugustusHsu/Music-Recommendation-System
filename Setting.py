#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:15:37 2019

@author: jim
"""

class setting(object):
    def __init__(self):
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
        self.save_path = './models/'
        
        #Song Model Setting
        self.song_size = 130009
        self.num_song = 5
        self.song_embedding_dim = 50
        self.song_rnn_size = 128
        self.song_num_rnn_layers = 1
        self.song_keep_probability = 0.8
        
        #Tag Model Setting
        self.tag_size = 16141
        self.num_tag = 3
        self.tag_embedding_dim = 25
        self.tag_rnn_size = 128
        self.tag_num_rnn_layers = 1
        self.tag_keep_probability = 0.8
        
        self.batch_size = 64
        self.fully_neuron=50
        self.output_dim=130009
        self.learning_rate=0.0017
        self.grad_clip=0.0
        #self.keep_probability=None
        self.epochs=100
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        