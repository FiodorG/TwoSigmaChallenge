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
import xgboost as xgb
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, is_valid_linkage
from itertools import chain
from pylab import rcParams
from sklearn.metrics import r2_score
from matplotlib.collections import LineCollection
from utilities import df_columns, add_features, r_score, clean_data_train, time_series_cv

np.random.seed(42)
pd.set_option('display.max_columns', 120)
np.set_printoptions(precision=5, suppress=True)


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

id = 500
df = df[df.id == id]

df = clean_data_train(df)
df = add_features(df)
columns = df_columns(df)

y_train = df[['y']]
y_train['y'] = df['y'] - df['y'].shift(1)
y_train = y_train.fillna(method='bfill')

scores = {}
sum_scores = {}
for column in columns:
    scores_for_column, periods = time_series_cv(df[[column]], y_train, number_folds=4, method='sliding')
    scores[column] = scores_for_column
    sum_scores[column] = np.average(scores_for_column)

top_sum_scores = dict(collections.Counter(sum_scores).most_common(40))
top_scores = { key: scores[key] for key in top_sum_scores.keys() }

scores_fold_1 = np.array(list(top_scores.values()))[:,0]
scores_fold_2 = np.array(list(top_scores.values()))[:,1]
scores_fold_3 = np.array(list(top_scores.values()))[:,2]

plt.figure(1)
plt.subplot(511)
plt.bar(np.arange(len(scores_fold_1)), scores_fold_1)
plt.ylabel('Fold1')
plt.subplot(512)
plt.bar(np.arange(len(scores_fold_2)), scores_fold_2)
plt.ylabel('Fold2')
plt.subplot(513)
plt.bar(np.arange(len(scores_fold_3)), scores_fold_3)
plt.ylabel('Fold3')
plt.subplot(514)
plt.plot(df.timestamp, y_train)
plt.ylabel('y')
for line in periods:
    plt.axvline(line[0], color='red', linestyle='-')
    plt.axvline(line[1], color='red', linestyle='-')
plt.subplot(515)
plt.plot(df.timestamp, np.cumsum(y_train))
plt.ylabel('cumsum(y)')
for line in periods:
    plt.axvline(line[0], color='red', linestyle='-')
    plt.axvline(line[1], color='red', linestyle='-')
plt.show()

print(list(top_scores.keys()))
 
#plot_all_columns_for_id_detail(df, id, columns=list(top_scores.keys()))

