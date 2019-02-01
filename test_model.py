#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:08:51 2019

@author: jim
"""

import tensorflow as tf
import numpy as np
import Setting
import Recommendation_Model

data_dir = 'data/'

idx_Song = np.load(data_dir + 'idx_Song.npy')
#idx_Song = idx_Song.reshape((idx_Song))
idx_Tag = np.load(data_dir + 'idx_Tag.npy')
print(idx_Song.shape)
print(idx_Tag.shape)

# shuffle the input data.
shuffle_indices = np.random.permutation(len(idx_Song))
idx_Song = idx_Song[shuffle_indices]
idx_Tag = idx_Tag[shuffle_indices]

Song_X = idx_Song[:360000,0:5]
print(Song_X.shape)
Song_Y = idx_Song[:360000,5:6]
print(Song_Y.shape)
Tag_X = idx_Tag[:360000,0:5,:]
print(Tag_X.shape)

Vali_Song_X = idx_Song[360000:400000,0:5]
print(Vali_Song_X.shape)
Vali_Song_Y = idx_Song[360000:400000,5:6]
print(Vali_Song_Y.shape)
Vali_Tag_X = idx_Tag[360000:400000,0:5,:]
print(Vali_Tag_X.shape)

Test_Song_X = idx_Song[400000:,0:5]
print(Test_Song_X.shape)
Test_Song_Y = idx_Song[400000:,5:6]
print(Test_Song_Y.shape)
Test_Tag_X = idx_Tag[400000:,0:5,:]
print(Test_Tag_X.shape)

setting = Setting.setting()

SongModel = Recommendation_Model.Recommendation_Model(setting)
SongModel.build_graph()
SongModel.train(Song_X, Tag_X, Song_Y, Vali_Song_X, Vali_Tag_X, Vali_Song_Y)
SongModel.save_model(model_name = 'Music_Recommendation')










