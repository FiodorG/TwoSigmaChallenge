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
#from utilities import r_score, df_columns, add_features
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble, decomposition)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler)
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
from utilities import r_score, df_columns, add_features, intra_cluster_correlation

np.random.seed(42)
pd.set_option('display.max_columns', 120)
np.set_printoptions(precision=5, suppress=True)


def get_distribution_repr(data):
    N = data.shape[0]
    nbBins = 200
    distribution_repr = np.zeros((N, nbBins))
    for i in range(0, N):
        hist, bin_edges = np.histogram(data[i, ], bins=nbBins, range=(-10, 10), density=True)
        distribution_repr[i, ] = hist * np.diff(bin_edges)

    return distribution_repr


def distance(df, theta=0.5):
    Ids = df['id'].unique()
    Ids = Ids[0:5]
    N = Ids.shape[0]
    result = np.zeros((N, N))
    for i in range(0, len(Ids)):
        df1 = df[df.id == Ids[i]]
        timestamp1 = df1['timestamp']
        for j in range(i+1, len(Ids)):
            df2 = df[df.id == Ids[j]]
            timestamp2 = df2['timestamp']
            timestamp1 = pd.Series(timestamp1).values
            timestamp2 = pd.Series(timestamp2).values
            index_time_series = list(set(timestamp1).intersection(timestamp2))
            My_df1 = df1.loc[df1['timestamp'].isin(index_time_series)]
            My_df2 = df2.loc[df2['timestamp'].isin(index_time_series)]
            Ts1 = pd.Series(My_df1['y']).values
            Ts2 = pd.Series(My_df2['y']).values
            T = len(index_time_series)

            distribution1 = get_distribution_repr(Ts1)
            distribution2 = get_distribution_repr(Ts2)
            d0 = np.sum(np.power((Ts1 - Ts2), 2)) / ((T * (T + 1) * (T - 1)) / 3)
            d1 = np.sum(np.power((np.sqrt(distribution1) - np.sqrt(distribution2)), 2)) / 2
            result[i, j] = theta * np.power(d0, 2) + (1 - theta) * np.power(d1, 2)
    return result


def get_distribution_repr_vector(df):
    Ids = df['id'].unique()
    result = []
    for i in range(0, len(Ids)):
        data = df[df.id == Ids[i]]
        data = pd.Series(data['y']).values
        N = data.shape[0]
        nbBins = 200
        distribution_repr = np.zeros((N, nbBins))
        for j in range(0, N):
            hist, bin_edges = np.histogram(data[j, ], bins=nbBins, range=(-10, 10), density=True)
            distribution_repr[j, ] = hist * np.diff(bin_edges)
        result.append(distribution_repr)
    return result


def fast_distance_computation(df, theta=0.5):
    Ids = list(df['id'].unique())
    N = Ids.shape[0]
    result = np.zeros((N, N))
    distribution = get_distribution_repr_vector(df=df)
    for i in range(0, len(Ids)):
        df1 = df[df.id == Ids[i]]
        timestamp1 = df1['timestamp']
        for j in range(i+1, len(Ids)):
            df2 = df[df.id == Ids[j]]
            timestamp2 = df2['timestamp']
            timestamp1 = pd.Series(timestamp1).values
            timestamp2 = pd.Series(timestamp2).values
            index_time_series = list(set(timestamp1).intersection(timestamp2))
            
            if (index_time_series == []) or (len(index_time_series) == 1):
                result[i, j] = 0
            else:
                My_df1 = df1.loc[df1['timestamp'].isin(index_time_series)]
                My_df2 = df2.loc[df2['timestamp'].isin(index_time_series)]
                Ts1 = pd.Series(My_df1['y']).values
                Ts2 = pd.Series(My_df2['y']).values
                T = len(index_time_series)
                distribution1 = distribution[i]
                distribution2 = distribution[j]
                dataframe_tmp1 = pd.DataFrame(data=distribution1, index=timestamp1)
                dataframe_tmp2 = pd.DataFrame(data=distribution2, index=timestamp2)
                distribution1 = dataframe_tmp1[dataframe_tmp1.index.isin(index_time_series)].values
                distribution2 = dataframe_tmp2[dataframe_tmp2.index.isin(index_time_series)].values
                d0 = np.sum(np.power((Ts1 - Ts2), 2)) / ((T * (T + 1) * (T - 1)) / 3)
                d1 = np.sum(np.power((np.sqrt(distribution1) - np.sqrt(distribution2)), 2)) / 2
                result[i, j] = theta * np.power(d0, 2) + (1 - theta) * np.power(d1, 2)
    return result


with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")
    
#df = df[df.id < 400]
df = df[['timestamp', 'id', 'y']]
df.fillna(0, inplace=True)

df3 = pd.pivot_table(df, values='y', index=['timestamp'], columns=['id'], aggfunc=np.sum)
cor = df3.corr()
cor.loc[:,:] =  np.tril(cor, k=-1)
cor = cor.stack()

distance = fast_distance_computation(df, theta=0.5)
distance2 = distance.T + distance

db = DBSCAN(eps=0.0001, min_samples=20, metric="precomputed").fit(distance2)
print(db.labels_)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
ids = df.id.unique()
toto = list(ids[np.where(labels == -1)])
intra_cluster_correlation(df, toto)

# convert the redundant n*n square matrix form into a condensed nC2 array
distArray = ssd.squareform(distance2) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
t = linkage(distArray, method='single', metric='euclidean')
t = linkage(distArray, method='single', metric='hamming')
t = linkage(distArray, method='complete', metric='euclidean')
t = linkage(distArray, method='average', metric='euclidean')
t = linkage(distArray, method='weighted', metric='euclidean')
t = linkage(distArray, method='centroid', metric='euclidean')
t = linkage(distArray, method='median', metric='euclidean')
t = linkage(distArray, method='ward', metric='euclidean')

print(is_valid_linkage(t))

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
scipy.cluster.hierarchy.dendrogram(
    t,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=15,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()