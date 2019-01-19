#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:08:51 2019

@author: jim
"""

import tensorflow as tf
import numpy as np
import Recommendation_Model

data_dir = 'data/'

idx_Song = np.load(data_dir + 'idx_Song.npy')
#idx_Song = idx_Song.reshape((idx_Song))
idx_Tag = np.load(data_dir + 'idx_Tag.npy')
print(idx_Song.shape)
print(idx_Tag.shape)

Song_X = idx_Song[:100000,0:5]
print(Song_X.shape)
Song_Y = idx_Song[:100000,5:6]
print(Song_Y.shape)
Tag_X = idx_Tag[:100000,0:5,:]
print(Tag_X.shape)

Vali_Song_X = idx_Song[-2000:,0:5]
print(Vali_Song_X.shape)
Vali_Song_Y = idx_Song[-2000:,5:6]
print(Vali_Song_Y.shape)
Vali_Tag_X = idx_Tag[-2000:,0:5,:]
print(Vali_Tag_X.shape)


tf.reset_default_graph()
SongModel = Recommendation_Model.Recommendation_Model(save_path='./models/', 
                                 mode='TRAIN', 
                                 song_size=130009, 
                                 num_song=5, 
                                 batch_size=64,
                                 rnn_size=5,
                                 num_rnn_layers=1,
                                 keep_probability=0.8,
                                 learning_rate=0.0017,
                                 epochs=100)
                      
SongModel.build_graph()
SongModel.train(Song_X, Song_Y, Vali_Song_X, Vali_Song_Y)












