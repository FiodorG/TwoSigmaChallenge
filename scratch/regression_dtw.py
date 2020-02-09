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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
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


def train_xgboost(df_train, df_validation, df_test, learning_rate, max_depth, reg_alpha, reg_lambda, rate_drop):
    x_train = df_train[top_columns]
    x_test = df_test[top_columns]
    x_validation = df_validation[top_columns]
    y_validation = df_validation[['y']]
    y_train = df_train[['y']]
    y_test = df_test[['y']]
    
    xgmat_train = xgb.DMatrix(x_train, label=y_train, feature_names=top_columns)
    xgmat_valid = xgb.DMatrix(x_validation, label=y_validation, feature_names=top_columns)
    xgmat_test = xgb.DMatrix(x_test, label=y_test, feature_names=top_columns)
    
    params_xgb = {
    'objective': 'reg:linear',
    'booster': 'dart',
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'subsample': 1,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
    'sample_type': 'weighted',
    'rate_drop': rate_drop,
    'eval_metric': 'rmse',
    'min_child_weight': 1,
    'base_score': 0,
    'normalize_type': 'forest'
    }
    
    num_rounds = 50
    bst = xgb.train(params_xgb, xgmat_train, num_rounds)
    params_xgb.update({'process_type': 'update', 'updater': 'refresh', 'refresh_leaf': False})
    bst_after = xgb.train(params_xgb, xgmat_valid, num_rounds, xgb_model=bst)
    
    #imp = pd.DataFrame(index=top_columns)
    #imp['train'] = pd.Series(bst.get_score(importance_type='gain'), index=top_columns)
    #imp['OOB'] = pd.Series(bst_after.get_score(importance_type='gain'), index=columns)
    #imp = imp.fillna(0)
    #ax = imp.sort_values('OOB').tail(10).plot.barh(title='Feature importances sorted by OOB', figsize=(7,4))
    
    y_predicted = bst_after.predict(xgmat_test, ntree_limit=num_rounds)
    return r_score(np.array(y_test), np.array(y_predicted)) * 100


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

df = remove_expired_ids(df)

id = 500
df = df[df.id == id]

df = clean_data(df)
df = add_y_inferred(df)
df = add_features(df)
columns = df_columns(df)

train_start = 0
train_end = 1000
test_start = 1000
test_end = 1400

df_train = df[(df.timestamp >= train_start) * (df.timestamp <= train_end)]
df_test = df[(df.timestamp >= test_start) * (df.timestamp < test_end)]

y = df_test.y.values
y_predictions = df_test.y.values * 0
y_predictions_last = df_test.y.values * 0

test_timestamps = list(df_test.timestamp.values)
for i in test_timestamps:
    print(i)
    df_interval = df[(df.timestamp > i - 60) * (df.timestamp <= i)]
        
    if ((i - test_timestamps[0]) % 30 == 0):
#        best_match, _, best_indices, y_predicted_last = find_closest_series(df_interval.y_cumsum_inferred.values, df_train.y_cumsum_inferred.values, verbose=False)
        best_match, _, best_indices, y_predicted_last = find_closest_series(df_interval.y_inferred.values, df_train.y_inferred.values, verbose=False)
        best_timestamps = df_train.timestamp.iloc[best_indices]
        df_train_best = df_train.iloc[best_timestamps]
#        y_predictions_last[i - test_timestamps[0]] = y_predicted_last

#    plt.figure(1)
#    plt.scatter(df_train_best.y_inferred.values, df_interval.y_inferred.values)
#    plt.show()
#    np.corrcoef(df_train_best.y_inferred.values, df_interval.y_inferred.values)[0, 1]
                           
#    plt.figure(1)
#    plt.plot(df_interval.y_cumsum_inferred.values, color='red')
##    plt.plot(np.cumsum(df[(df.timestamp < 2000)].y), color='green')
#    plt.plot(df_train.iloc[best_timestamps].y_cumsum_inferred.values, color='blue')
#    plt.show()
#
#    plt.figure(1)
#    plt.plot(df_interval.y_inferred.values, color='red')
##    plt.plot(np.cumsum(df[(df.timestamp < 2000)].y), color='green')
#    plt.plot(df_train.iloc[best_timestamps].y_inferred.values, color='blue')
#    plt.show()

    scores = {}
    for column in columns:
        model = linear_model.Ridge(normalize=True)
        model.fit(df_train_best[[column]], df_train_best[['y']])
        y_predicted = model.predict(df_interval[[column]])
        scores[column] = r_score(df_interval.y_inferred.shift(-1).values[:-1], y_predicted[:-1]) * 100

    top_scores = dict(collections.Counter(scores).most_common(20))
    top_scores = { key: scores[key] for key in top_scores.keys() }
    variables_to_keep = list(top_scores.keys())

    df_train_final = df_train_best.append(df_interval, ignore_index=True)
    df_train_final = df_train_final[['timestamp'] + variables_to_keep + ['y', 'y_inferred']]
    df_test_final = df_train_final[-1:]
    df_train_final = df_train_final[:-1]
    
#    plt.figure(1)
#    plt.scatter(df_train_final.y_inferred.values, df_train_final.technical_40_diff1.values)
#    plt.show()
#    np.corrcoef(df_train_final.y_inferred.values, df_train_final.technical_40_diff1.values)[0, 1]

    predictions_final = {}
    for column in variables_to_keep:
        model = linear_model.Ridge(normalize=True)
        model.fit(df_train_final[[column]].values[:-1], df_train_final[['y_inferred']].shift(-1).values[:-1])
        y_predicted = model.predict(df_test_final[[column]])
        predictions_final[column] = y_predicted[0][0]

    y_predictions[i - test_timestamps[0]] = np.median([ predictions_final[key] for key in predictions_final.keys() ])


plt.figure(1)
plt.plot(y, color='red')
#plt.plot(df[(df.timestamp < 1000)].y.values, color='green')
plt.plot(y_predictions, color='blue')
plt.plot(y_predictions_last, color='green')
plt.show()

print(r_score(y, y_predictions) * 100)
