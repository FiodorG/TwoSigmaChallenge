def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
from sklearn import (svm, linear_model, preprocessing, ensemble)
from libraries import kagglegym
from sklearn.cluster import *
import pandas as pd
import scipy.signal
import random
from libraries import visualization
from sklearn import *
from sklearn.feature_selection import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import *
from hmmlearn import hmm
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import *
import timeit
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import xgboost as xgb
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import *
from itertools import chain
from pylab import rcParams
from sklearn.metrics import r2_score
from matplotlib.collections import LineCollection
#from utilities import df_columns, add_features, r_score, clean_data, time_series_cv, add_y_inferred
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import sys
from math import *

np.random.seed(42)
pd.set_option('display.max_columns', 120)
np.set_printoptions(precision=5, suppress=True)

col = 'technical_5'

df_temp = df[['timestamp','id','y',col]]
df_temp = df_temp[df_temp.id==500]
df_temp['y_cumsum'] = np.cumsum(df_temp.y)

df_temp = df_temp[df_temp[col] != 0]


#plt.plot(df_temp.timestamp, df_temp.technical_20, 'r')
#plt.plot(df_temp.timestamp, np.cumsum(df_temp.y), 'b')

df = df_train
col = 'trend_states'
df[col].plot(label=str(col), legend=True)
df.y_inferred_cumsum_ma_40.plot(secondary_y=True, label="y_cumsum", legend=True)
df.y_inferred_cumsum_ma_80.plot(secondary_y=True, label="y_ma40", legend=True)
df.y_cumsum.plot(secondary_y=True, label="y_ma80", legend=True)

df = df_validation
col = 'trend_states'
df[col].plot(label=str(col), legend=True)
df.y_inferred_cumsum_ma_40.plot(secondary_y=True, label="y_cumsum", legend=True)
df.y_inferred_cumsum_ma_80.plot(secondary_y=True, label="y_ma40", legend=True)
df.y_cumsum.plot(secondary_y=True, label="y_ma80", legend=True)

df = df_test
col = 'trend_states'
df[col].plot(label=str(col), legend=True)
df.y_inferred_cumsum_ma_40.plot(secondary_y=True, label="y_cumsum", legend=True)
df.y_inferred_cumsum_ma_80.plot(secondary_y=True, label="y_ma40", legend=True)
df.y_cumsum.plot(secondary_y=True, label="y_ma80", legend=True)
#df.y.plot(secondary_y=True, label="y", legend=True)


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

mean_values = df.mean(axis=0)
df.fillna(mean_values, inplace=True)
#id = 505
#df = df[df.id == id]
df = add_y_inferred(df)

def ewm_mean(x,span_in):
    return(x.ewm(span=span_in).mean())

#Build EWM parameters
df['EWM_26_mean']  = df.groupby('id')['y'].apply(lambda x: ewm_mean(x,span_in=26))
df['y_shifted'] = df.groupby('id')['y'].shift(1).fillna(0)
df['y_shifted_cumsum'] = df.groupby('id')['y'].shift(1).fillna(0).apply(np.cumsum)
df['EWM_26_mean_s'] = df.groupby('id')['y_shifted'].apply(lambda x: ewm_mean(x,span_in=26))

df_train = df[df.timestamp < 1000]
df_test = df[df.timestamp >= 1000]

ewm_features = ['technical_30','technical_13','technical_20','technical_21','technical_19','technical_17','technical_11','technical_2']
model = GradientBoostingRegressor(loss='ls', max_depth=5, learning_rate=0.05)
model.fit(X=df_train[ewm_features],y=df_train['EWM_26_mean_s'])
df_train['EWM_26s_pred'] = model.predict(X=df_train[ewm_features]) 
df_test['EWM_26s_pred'] = model.predict(X=df_test[ewm_features])

def ewm_reverse(data,span=26):
    alpha = 2/(span+1)
    return (data-(1-alpha)*data.shift(1).fillna(0))/alpha

# Inverse transform
df_train['yEWM_26'] = df_train.groupby('id')['EWM_26s_pred'].apply(lambda x: ewm_reverse(x, span=26))
df_test['yEWM_26'] = df_test.groupby('id')['EWM_26s_pred'].apply(lambda x: ewm_reverse(x, span=26))
df_test['yEWM_26_cumsum'] = df_test.groupby('id')['EWM_26s_pred'].apply(np.cumsum)

find_R_value(df_test['yEWM_26'], df_test.groupby('id')['y'].shift(1).fillna(0))

def find_R_value(y_predict,y_actual):
    mean = np.mean(y_actual)
    Rrr = 1 - sum((y_predict-y_actual)**2)/sum((y_actual-mean)**2)
    return np.sign(Rrr)*np.sqrt(np.abs(Rrr))  

df = df_test[df_test.id == 500]

df.yEWM_26_cumsum.plot(secondary_y=False, label="y_ma40", legend=True)
df.y_shifted_cumsum.plot(secondary_y=False, label="y_ma80", legend=True)

df.y_inferred_cumsum.plot(secondary_y=False, label="y_ma40", legend=True)
df.y_cumsum.plot(secondary_y=False, label="y_ma80", legend=True)

df.y_inferred.plot(secondary_y=False, label="y_ma40", legend=True)
df.y_shifted.plot(secondary_y=False, label="y_ma80", legend=True)

print(find_R_value(df.y_inferred_cumsum,df.y_cumsum.shift(1)))
print(find_R_value(df.y_inferred,df.y_shifted))

#plt.show()
