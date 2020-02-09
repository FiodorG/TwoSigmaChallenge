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
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import scipy.signal
import random
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble, decomposition, manifold, covariance, cluster)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from hmmlearn import hmm
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
import timeit
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#import xgboost as xgb
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, is_valid_linkage
from itertools import chain
from pylab import rcParams
from utilities import r_score, df_columns, add_features, add_y_inferred, add_states, clean_data

np.random.seed(42)
pd.set_option('display.max_columns', 120)
np.set_printoptions(precision=5, suppress=True)


def plot_states(df, col='y_inferred_cumsum'):
    df = add_states(df, col=col, method='vol', window=20, minimum_length=40, states=2)
    df = add_states(df, col=col, method='trend', window=20, minimum_length=40, states=2)

#    plt.figure(1)
#    plt.plot(df[col+'_ma_40'])
#    plt.plot(df[col+'_ma_120'])
#    plt.plot(df.states / 10)
#    plt.plot(df.y_cumsum)
#    plt.show()

    y_states_vol_high = df[df.vol_states == 1][['y']].values
    y_states_vol_low = df[df.vol_states == 0][['y']].values
    y_states_trend_pos = df[df.trend_states == 1][['y']].values
    y_states_trend_neg = df[df.trend_states == 0][['y']].values

    plt.figure(4)
    plt.subplot(211)
    plt.title('Histogram of High Vol')
    plt.hist(y_states_vol_high, bins=100, range=(-0.08, 0.08), color='red')
    plt.subplot(212)
    plt.title('Histogram of Low Vol')
    plt.hist(y_states_vol_low, bins=100, range=(-0.08, 0.08), color='blue')
    plt.figure(5)
    plt.subplot(211)
    plt.title('Histogram of Positive Trend')
    plt.hist(y_states_trend_pos, bins=100, range=(-0.08, 0.08), color='red')
    plt.subplot(212)
    plt.title('Histogram of Negative Trend')
    plt.hist(y_states_trend_neg, bins=100, range=(-0.08, 0.08), color='blue')
    plt.show()

#    y_states_trend_neg_high_vol = df[df['trend_states'] == 0]
#    y_states_trend_neg_high_vol = y_states_trend_neg_high_vol[y_states_trend_neg_high_vol['vol_states'] == 1 ][['y']].values
#
#    y_states_trend_neg_low_vol = df[df['trend_states'] == 0]
#    y_states_trend_neg_low_vol = y_states_trend_neg_low_vol[y_states_trend_neg_low_vol['vol_states'] == 0 ][['y']].values
#
#    y_states_trend_pos_high_vol = df[df['trend_states'] == 1]
#    y_states_trend_pos_high_vol = y_states_trend_pos_high_vol[y_states_trend_pos_high_vol['vol_states'] == 1 ][['y']].values
#
#    y_states_trend_pos_low_vol = df[df['trend_states'] == 1]
#    y_states_trend_pos_low_vol = y_states_trend_pos_low_vol[y_states_trend_pos_low_vol['vol_states'] == 0 ][['y']].values
#
#    plt.figure(1)
#    plt.subplot(211)
#    plt.hist(y_states_trend_neg_high_vol, bins=100, range=(-0.08, 0.08), color='red')
#    plt.subplot(212)
#    plt.hist(y_states_trend_neg_low_vol, bins=100, range=(-0.08, 0.08), color='blue')
#    plt.figure(2)
#    plt.subplot(211)
#    plt.hist(y_states_trend_pos_high_vol, bins=100, range=(-0.08, 0.08), color='red')
#    plt.subplot(212)
#    plt.hist(y_states_trend_pos_low_vol, bins=100, range=(-0.08, 0.08), color='blue')
#    plt.show()


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

df = df[df.id==500]
df = clean_data(df)
df = add_features(df)

df = add_y_inferred(df)
columns = df_columns(df)

df = add_states(df, col='y_inferred_cumsum', method='vol', window=60, minimum_length=40, states=2)
df = add_states(df, col='y_inferred_cumsum', method='trend', window=60, minimum_length=40, states=2)

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(df.trend_states.values / 100, label='rolling_vol', color='red')
#ax.plot(df.rolling_vol, label='rolling_vol', color='green')
#ax2 = ax.twinx()
#ax2.plot(df.y_cumsum, label='y_cumsum', color='blue')
#plt.show()

df.fillna(0, inplace=True)
df_train = df[df.timestamp < 1000][df.trend_states == 0]
df_test = df[df.timestamp >= 1000][df.trend_states == 0]

scores = {}
betas = {}
intercept = {}
predictions = {} 
for column in columns:
    
    model = linear_model.LinearRegression(normalize=True)
    model.fit(df_train[[column]], df_train[['y']])
    y_predicted = model.predict(df_test[[column]])
    predictions[column] = y_predicted
    score = r_score(df_test.y.values, y_predicted) * 100            
    scores[column] = score
    betas[column] = model.coef_[0][0]
    intercept[column] = model.intercept_
    

top_scores = dict(collections.Counter(scores).most_common(40))
betas = { key: betas[key] for key in top_scores.keys() }
predictions = { key: predictions[key] for key in top_scores.keys() }
top_scores = { key: scores[key] for key in top_scores.keys() }
t = np.array(list(top_scores.values()))

plt.figure(1)
plt.subplot(111)
plt.bar(np.arange(len(t)), t)

top_scores = list(dict(collections.Counter(scores).most_common(3)).keys())
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ax.plot(df[top_scores[0]].values, label=top_scores[0], color='red')
ax2 = ax.twinx()
ax2.plot(df.y_cumsum, label='y_cumsum', color='blue')
ax2.plot(df.trend_states.values)
plt.title(top_scores[0])
ax = fig.add_subplot(3, 1, 2)
ax.plot(df[top_scores[1]].values, label=top_scores[1], color='red')
#ax.plot(df.vol_states.values/100)
ax2 = ax.twinx()
ax2.plot(df.trend_states.values)
ax2.plot(df.y_cumsum, label='y_cumsum', color='blue')
plt.title(top_scores[1])
ax = fig.add_subplot(3, 1, 3)
ax.plot(df[top_scores[2]].values, label=top_scores[2], color='red')
#ax.plot(df.vol_states.values/100)
ax2 = ax.twinx()
ax2.plot(df.trend_states.values)
ax2.plot(df.y_cumsum, label='y_cumsum', color='blue')
plt.title(top_scores[2])
plt.show()

y = scipy.signal.savgol_filter(df_test.y, 51, 3)
top_scores = list(dict(collections.Counter(scores).most_common(3)).keys())
fig = plt.figure(1)
ax = fig.add_subplot(3, 1, 1)
ax.plot( df_test.timestamp,y, color='red')
ax2 = ax.twinx()
ax.plot( df_test.timestamp, predictions[top_scores[0]], color='blue')
plt.title(top_scores[0])
ax = fig.add_subplot(3, 1, 2)
ax.plot( df_test.timestamp,y, color='red')
ax2 = ax.twinx()
ax.plot( df_test.timestamp, predictions[top_scores[1]], color='blue')
plt.title(top_scores[1])
ax = fig.add_subplot(3, 1, 3)
ax.plot( df_test.timestamp,y, color='red')
ax2 = ax.twinx()
ax.plot( df_test.timestamp, predictions[top_scores[2]], color='blue')
plt.title(top_scores[2])
plt.show()



#plot_states(df, col='y_inferred_cumsum')
#plot_states(df, col='y_cumsum')

#plt.figure(1)
#plt.plot(df['y'], df['y_lag'], 'o')
#plt.show()

#df.y_fit_cumsum.plot(label=str('y_lag_cumsum'), legend=True)
#df.y_cumsum.plot(secondary_y=False, label='y_cumsum', legend=True)

## HMM
#lag = 20
#states = 2
#rolling_vol1 = df[['y']].rolling(lag).std()
#y1 = df[['y']]
#ycumsum1 = np.cumsum(df[['y']])
#rolling_vol1 = rolling_vol1.dropna(axis=0)
#model1 = hmm.GaussianHMM(n_components=states, covariance_type="full", algorithm='viterbi')
#model1.fit( rolling_vol1 )
#states1 = model1.predict( rolling_vol1 )
#
#rolling_vol2 = df[['y_fit']].rolling(lag).std()
#y2 = df[['y_fit']]
#rolling_vol2 = rolling_vol2.dropna(axis=0)
#model2 = hmm.GaussianHMM(n_components=states, covariance_type="full", algorithm='viterbi')
#model2.fit( rolling_vol2 )
#states2 = model2.predict( rolling_vol2 )
#
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
##ax.plot(df['y'])
##ax = fig.add_subplot(3, 1, 2)
#ax.plot(rolling_vol1, label='rolling_vol1', color='red')
#ax.plot(y1, label='rolling_vol1', color='green')
#ax2 = ax.twinx()
#ax2.plot(ycumsum1, label='rolling_vol1', color='black')
#ax.plot(rolling_vol1.index, states1 / 100, label='states1', color='red')
#ax.plot(rolling_vol2, label='rolling_vol2', color='blue')
#ax.plot(rolling_vol2.index, states2 / 100, label='states2', color='blue')
##ax.plot(rolling_vol1.index, (states1 - states2) / 100, label='statesdiff', color='green')
##ax = fig.add_subplot(3, 1, 3)
##ax.plot(np.cumsum(df['y']))
#plt.show()