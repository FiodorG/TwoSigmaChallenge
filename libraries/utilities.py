def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import collections
import statsmodels
import sklearn
import sys
import timeit
from math import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.linear_model import *
from hmmlearn import *
from sklearn.pipeline import *
from sklearn.grid_search import *
from sklearn.learning_curve import *
from sklearn.model_selection import *
from sklearn.manifold import *
from sklearn.metrics import *
from scipy.cluster.hierarchy import *
from matplotlib.collections import *
from sklearn.decomposition import *
from sklearn.cross_decomposition import *
from fastdtw import *


np.random.seed(42)
pd.set_option('display.max_columns', 51)
np.set_printoptions(precision=5, suppress=True)


#########################################################
def remove_expired_ids(df):
    df_timestamps = df[['id', 'timestamp']].groupby('id').agg({'timestamp': [np.max]})
    ids = list(df_timestamps[df_timestamps.timestamp.amax == df.timestamp.max()].index)
    df = df[df.id.isin(ids)]
    return df


#########################################################
def find_ids_with_all_timestamps(df):
    df_timestamps = df[['id', 'timestamp']].groupby('id').agg({'timestamp': [np.min, np.max]})
    ids = list(df_timestamps[(df_timestamps.timestamp.amax == df.timestamp.max())*(df_timestamps.timestamp.amin == df.timestamp.min())].index)
    return ids


#########################################################
def binary_features():
    feat1 = ['technical_2', 'technical_6', 'technical_10', 'technical_11', 'technical_14', 'technical_17', 'technical_29', 'technical_43']
    feat2 = ['technical_0', 'technical_9', 'technical_12', 'technical_18', 'technical_32', 'technical_37', 'technical_38', 'technical_39']
    feat3 = ['technical_16', 'technical_42']
    feat4 = ['technical_22', 'technical_34']

    return [feat1, feat2, feat3, feat4]


#########################################################
def clean_data(df, columns=[]):

    for feat in binary_features():
        if 'technical_2' in feat:
            df[df[feat].apply(lambda x: x > -1)] = 0
            df[df[feat].apply(lambda x: x <= -1)] = -2
        elif 'technical_0' in feat:
            df[df[feat].apply(lambda x: x > -0.5)] = 0
            df[df[feat].apply(lambda x: x <= -0.5)] = -1
        elif 'technical_16' in feat:
            df[df[feat].apply(lambda x: x >= 0.5)] = 1
            df[df[feat].apply(lambda x: x <= -0.5)] = -1
            df[(df[feat].apply(lambda x: x > -0.5)) & (df[feat].apply(lambda x: x < 0.5))] = 0
        else:
            df[df[feat].apply(lambda x: x >= 0.25)] = 0.5
            df[df[feat].apply(lambda x: x <= -0.25)] = -0.5
            df[(df[feat].apply(lambda x: x > -0.25)) & (df[feat].apply(lambda x: x < 0.25))] = 0

    if columns == []:
        columns = df_columns(df)

    df = df[['id', 'timestamp'] + columns + ['y']]
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.median(axis=0))

#    df = df[df.technical_20 != 0]
    low_y_cut = np.percentile(df.y, 5)
    high_y_cut = np.percentile(df.y, 95)
    y_is_above_cut = (df.y > high_y_cut)
    y_is_below_cut = (df.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    df = df[y_is_within_cut]

    return df


#########################################################
def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r


#########################################################
def df_columns(df, which='all'):

    if which == 'all':
        return list(df.columns.drop(['y', 'timestamp', 'id', 'y_cumsum'] + sum(binary_features()), errors='ignore'))
    elif which == 'base':
        columns_derived = [c for c in df.columns if 'derived' in c]
        columns_fundamental = [c for c in df.columns if 'fundamental' in c]
        columns_technical = [c for c in df.columns if 'technical' in c]
        all_columns = columns_derived + columns_fundamental + columns_technical
        return all_columns


#########################################################
def add_features(df, columns=[]):
    if columns == []:
        columns = df_columns(df)

    for col in columns:
        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_20'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_200'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_std_20'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_std_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_std_200'}), on = ['timestamp','id'])

        df[col+'_lag1'] = df[[col]].shift(1)
        df[col+'_lag2'] = df[[col]].shift(2)
        df[col+'_lag4'] = df[[col]].shift(4)

        df[col+'_diff1'] = df[col] - df[col].shift(1)
        df[col+'_diff2'] = df[col] - df[col].shift(2)
        df[col+'_diff5'] = df[col] - df[col].shift(5)
        df[col+'_diff10'] = df[col] - df[col].shift(10)
        df[col+'_diff20'] = df[col] - df[col].shift(20)
        df[col+'_diff40'] = df[col] - df[col].shift(40)
        
        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).quantile(0.9))
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_percentile_20_90'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).quantile(0.9))
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_percentile_60_90'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).quantile(0.6))
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_percentile_20_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).quantile(0.6))
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_percentile_60_60'}), on = ['timestamp','id'])

    df = df.fillna(method='bfill')

    return df


#########################################################
def intra_cluster_correlation(df, indices):

    df_cluster = df[df.id.isin(indices)][['timestamp','id','y']].pivot('timestamp', 'id')
    corr = df_cluster.corr()
    dist = corr.as_matrix()
    plt.hist(dist.flatten(), bins=100)
    plt.title("Correlation between Ids")
    sns.clustermap(dist, metric="euclidean", method="average")


#########################################################
def add_y_inferred(df, alpha = 0.9327, verbose=False):

    min_y, max_y = min(df.y), max(df.y)
    df['y_cumsum_inferred'] = df['technical_20'] - df['technical_30'] + df['technical_13']
    df['y_inferred'] = ((df['y_cumsum_inferred'] - df.groupby('id')['y_cumsum_inferred'].shift(1).fillna(0).values * alpha) / (1 - alpha)).clip(min_y,max_y).fillna(0)
#    df['y_inferred'] = df.groupby('id')['y_inferred'].shift(-1).fillna(0)
    df['y_cumsum_inferred'] = df.groupby('id')['y_inferred'].apply(np.cumsum)

    if verbose:
        df['y_lag'] = df.groupby('id')['y'].shift(1).fillna(0)
        df['y_cumsum'] = df.groupby('id')['y'].apply(np.cumsum)
        df['y_diff'] = df['y'] - df['y_inferred']
        print('Number points within 0.0005 of true y: {}'.format(np.sum(abs(df['y_diff']) < 0.0005)))
        plt.hist(df['y_diff'],bins=200)
        plt.grid()
        plt.show()
        n0s = 10000
        plt.scatter(df['y_inferred'][:n0s], df['y'][:n0s], alpha=0.1)
        plt.grid()
        plt.show()
        df.y.plot(legend=True)
        df.y_inferred.plot(legend=True)
        df.y_cumsum.plot(legend=True)
        df.y_cumsum_inferred.plot(legend=True)

    return df


#########################################################
def clean_states(df, minimum_length=40):
    
    index = df.first_valid_index()
    reference_value = df[index]
    count = 1
    for i in range(index, len(df)):
        if reference_value != df[i]:
            reference_value = df[i]
            if count < minimum_length:
                for j in range((i-count), i):
                    df[j] = reference_value
                    count += minimum_length
            else:
                count = 1
        else:
            count += 1
    return df


#########################################################
def add_states(df, col='y_inferred', method='vol', window=20, minimum_length=40, states=2):

    if method == 'vol':
        rolling_vol = df[[col]].rolling(window).std()
        rolling_vol = rolling_vol.fillna(method='bfill')
        model = hmm.GaussianHMM(n_components=states, covariance_type="full", algorithm='viterbi')
        model.fit(rolling_vol)

        df['rolling_vol'] = rolling_vol
        df[method+'_states'] = model.predict(rolling_vol)

    elif method == 'trend':
        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(40).mean()).fillna(method='bfill')
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_40'}), on = ['timestamp','id'])
        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(80).mean()).fillna(method='bfill')
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_80'}), on = ['timestamp','id'])

        window = 51
        df[method+'_states'] = 1 * (scipy.signal.savgol_filter(df[col+'_ma_40'], window, 3) > scipy.signal.savgol_filter(df[col+'_ma_80'], window, 3))

    else:
        raise ValueError('method unknown')

    df[method+'_states'] = clean_states(df[method+'_states'], minimum_length=minimum_length)

    return df
