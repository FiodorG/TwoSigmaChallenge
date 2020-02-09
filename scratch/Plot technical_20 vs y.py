# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:30:13 2017

@author: Fiodor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble)
import scipy.signal

with pd.HDFStore("../input/train.h5", "r") as data_file:
    df_train = data_file.get("train")

#df_train.fillna(df_train.mean(axis=0), inplace=True)

#df_train_t = df_train[df_train.id==600][['timestamp','id','technical_20','y']]
#plt.figure(1)
#plt.plot(df_train_t.technical_20, df_train_t.y)
#plt.show()

df_train_y = df_train[df_train.technical_20 != 0][df_train.id==600][['timestamp','technical_20','y']]

y = df_train_y.y
x = df_train_y.technical_20
                      
plt.figure(1)
#plt.plot(x, y)

# fit with np.polyfit
m, b = np.polyfit(x, y, 1)

plt.plot(np.diff(np.power(x,1),2), np.diff(y,2), 'o')
#plt.plot(x, m*x + b, '-')

plt.show()
