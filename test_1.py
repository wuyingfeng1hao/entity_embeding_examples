# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:06:34 2020
refer: https://github.com/wuyingfeng1hao/deepschool.io/blob/master/DL-Keras_Tensorflow/Lesson%2006%20-%20contraception.ipynb
@author: WON4SZH
"""

import pandas as pd
import numpy as np

from keras.models import Sequential

import keras
#note: https://stackoverflow.com/questions/56315726/cannot-import-name-merge-from-keras-layers
print(keras.__version__)

import os
os.getcwd()

wd = "C:\\Users\\WON4SZH\\Desktop\\report to WangDong\\008_book_tasks\\2020_如何突破传统机器学习的瓶颈\\code\\entity_examples\\entity_embeding_examples"
os.chdir(wd)

from keras.layers import Dense, Activation, Embedding,  Flatten #Merge,

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

df = pd.read_csv('cmc.data',header=None,names=['Age','Education','H_education',
                                                     'num_child','Religion', 'Employ',
                                                     'H_occupation','living_standard',
                                                     'Media_exposure','contraceptive'])


df.head()
df.shape


df.isnull().any()

df.Education.hist()

df.contraceptive.hist()

df.dtypes

def one_hot_encoding(idx):
    y = np.zeros((len(idx),max(idx)+1))
    y[np.arange(len(idx)), idx] = 1
    return y

scaler = StandardScaler()
df[['Age','num_child']] = scaler.fit_transform(df[['Age','num_child']])

x = df[['Age','num_child','Employ','Media_exposure']].values

y = one_hot_encoding(df.contraceptive.values-1)

liv_cats = df.living_standard.max()
edu_cats = df.Education.max()

liv = df.living_standard.values - 1
liv_one_hot = one_hot_encoding(liv)
edu = df.Education.values - 1
edu_one_hot = one_hot_encoding(edu)

train_x, test_x, train_liv, \
test_liv, train_edu, test_edu, train_y, test_y = train_test_split(x,liv_one_hot,edu_one_hot,y,test_size=0.1, random_state=1)

train_x = np.hstack([train_x, train_edu, train_liv])
test_x = np.hstack([test_x, test_edu, test_liv])

train_x.shape

model = Sequential()

#update to keras2 API for Dense function.
model.add(Dense(input_dim=train_x.shape[1],units=12))

model.add(Activation('relu'))

model.add(Dense(units=3))

model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, nb_epoch=100, verbose=2)

model.summary()

for w in model.get_weights():
    print(w.shape)
    
model.evaluate(test_x, test_y, batch_size=256)