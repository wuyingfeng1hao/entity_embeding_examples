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

from keras.layers import Dense, Activation, Embedding,  Flatten #Merge,

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
