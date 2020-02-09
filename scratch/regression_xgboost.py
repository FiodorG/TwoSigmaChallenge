def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
from libraries import kagglegym
from sklearn.cluster import *
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
from sklearn.metrics import *
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import xgboost as xgb
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import *
from itertools import chain
from pylab import rcParams
from matplotlib.collections import LineCollection
#from utilities import df_columns, add_features, r_score, clean_data, time_series_cv, add_y_inferred
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import sys
from math import *
import statsmodels.formula.api as smf

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
def plot_three_best_predictions(df_test, scores, predictions, filter_window=0):
    y = df_test.y
    
    if filter_window > 0:
        y = scipy.signal.savgol_filter(df_test.y, 51, 3)
        
    top_scores = list(dict(collections.Counter(scores).most_common(3)).keys())
    fig = plt.figure(1)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot( df_test.timestamp, y, color='red')
    ax2 = ax.twinx()
    ax.plot( df_test.timestamp, predictions[top_scores[0]], color='blue')
    plt.title(top_scores[0])
    ax = fig.add_subplot(3, 1, 2)
    ax.plot( df_test.timestamp, y, color='red')
    ax2 = ax.twinx()
    ax.plot( df_test.timestamp, predictions[top_scores[1]], color='blue')
    plt.title(top_scores[1])
    ax = fig.add_subplot(3, 1, 3)
    ax.plot( df_test.timestamp, y, color='red')
    ax2 = ax.twinx()
    ax.plot( df_test.timestamp, predictions[top_scores[2]], color='blue')
    plt.title(top_scores[2])
    plt.show()


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

id = 500
df = df[df.id == id]

df = clean_data(df)
df = add_y_inferred(df)
df = add_features(df)
df = add_states(df, col='y_inferred_cumsum', method='trend', window=60, minimum_length=50, states=2)
columns = df_columns(df)

train_start = 0
train_end = 1000
validation_start = 801
validation_end = 1000
test_start = 1001
test_end = 2000

state_name = 'trend_states'
df_train = df[(df.timestamp >= train_start) * (df.timestamp < train_end)][df[state_name] == 1]
df_test = df[(df.timestamp >= test_start) * (df.timestamp < test_end)][df[state_name] == 1]
df_validation = df[(df.timestamp >= validation_start) * (df.timestamp < validation_end)][df[state_name] == 1]
df_train_validation = df[(df.timestamp >= train_start) * (df.timestamp < train_end) + (df.timestamp >= validation_start) * (df.timestamp < validation_end)][df[state_name] == 1]

df_train = df[(df.timestamp >= train_start) * (df.timestamp < train_end)][df[state_name] == 0]
df_test = df[(df.timestamp >= test_start) * (df.timestamp < test_end)][df[state_name] == 0]
df_validation = df[(df.timestamp >= validation_start) * (df.timestamp < validation_end)][df[state_name] == 0]
df_train_validation = df[(df.timestamp >= train_start) * (df.timestamp < train_end) + (df.timestamp >= validation_start) * (df.timestamp < validation_end)][df[state_name] == 0]

scores = {}
sum_scores = {}
for column in columns:
    scores_for_column, periods = time_series_cv(df_train[[column]], df_train.y, number_folds=3, method='increasing')
    scores[column] = scores_for_column
    sum_scores[column] = np.average(scores_for_column)

top_sum_scores = dict(collections.Counter(sum_scores).most_common(40))
top_scores = { key: scores[key] for key in top_sum_scores.keys() }

#plot_three_best_predictions(df_test, scores, predictions, filter_window=0)

top_columns = list(top_scores.keys())

#pca = PCA(n_components=10, whiten=True)
#pca.fit(df_train[top_columns])
#explained_variance = np.cumsum(pca.explained_variance_ratio_)
#number_of_components = np.where(explained_variance > 0.95)[0][0]
#pca = PCA(n_components=number_of_components, whiten=True)
#x_train = pca.fit_transform(df_train[top_columns])
#x_test = pca.fit_transform(df_test[top_columns])

##

x_train = df_train[top_columns]
x_test = df_test[top_columns]
x_validation = df_validation[top_columns]
y_validation = df_validation[['y']]
y_train = df_train[['y']]
y_test = df_test[['y']]

#model = PLSRegression(n_components=2, scale=False, max_iter=1000, tol=1e-03)
#model.fit(x_train, y_train)
#y_predicted = model.predict(x_test)
#
#print(r_score(np.array(y_test), np.array(y_predicted)) * 100)
#plt.figure(1)
#plt.plot(np.array(scipy.signal.savgol_filter(y_test.y, 41, 3)), color='red')
#plt.plot(np.array(y_predicted), color='blue')
#plt.show()
#
#plt.figure(1)
#plt.plot(np.cumsum(df[df.id == id].y), color='blue')
#plt.show()


#########################################################
learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9, 2]
max_depths = [1, 2, 3, 4]
reg_alphas = [1, 0.1, 0.01]
reg_lambdas = [1, 0.1, 0.01]
rate_drops = [0.1, 0.3, 0.5, 0.7, 0.9]

results = {}
for learning_rate in learning_rates:
    for max_depth in max_depths:
        for reg_alpha in reg_alphas:
            for reg_lambda in reg_lambdas:
                for rate_drop in rate_drops:
                    results[str(learning_rate)+', '+
                            str(max_depth)+', '+
                            str(reg_alpha)+', '+
                            str(reg_lambda)+', '+
                            str(rate_drop)]  \
                    = train_xgboost(df_train, df_validation, df_test, learning_rate, max_depth, reg_alpha, reg_lambda, rate_drop)

top_results = dict(collections.Counter(results).most_common(40))

plt.figure(1)
plt.plot(np.array(scipy.signal.savgol_filter(y_test.y, 101, 3)), color='red')
plt.plot(np.array(y_predicted), color='blue')
plt.show()