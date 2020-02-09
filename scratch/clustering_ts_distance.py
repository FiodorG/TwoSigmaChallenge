def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
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

np.random.seed(42)


#########################################################
def get_distribution(x, nbBins):
    hist, bin_edges = np.histogram(x, bins=nbBins, range=(-10, 10), density=True)
    return hist * np.diff(bin_edges)


#########################################################
def get_distribution_repr(data):
    nbBins = 100
    return np.array([get_distribution(x, nbBins) for x in data])


#########################################################
def distance(df, theta=0.5):
    df = df[['timestamp', 'id', 'y']]
    Ids = df['id'].unique()
    N = Ids.shape[0]
    result = np.zeros((N, N))

    for index, item in enumerate(Ids):

        print(index)
        df1 = df[df.id == item]
        timestamp1 = np.array(df1['timestamp'])

        for index2, item2 in enumerate(Ids):

            df2 = df[df.id == item2]
            timestamp2 = np.array(df2['timestamp'])
            index_time_series = list(set(timestamp1).intersection(timestamp2))

            My_df1 = df1[df1['timestamp'].isin(index_time_series)]
            My_df2 = df2[df2['timestamp'].isin(index_time_series)]

            Ts1 = np.array(My_df1['y'])
            Ts2 = np.array(My_df2['y'])

            T = len(index_time_series)
            Distribution1 = get_distribution_repr(Ts1)
            Distribution2 = get_distribution_repr(Ts2)

            d0 = np.sum(np.power((Ts1 - Ts2), 2)) / ((T * (T + 1) * (T - 1)) / 3)
            d1 = np.sum(np.power((np.sqrt(Distribution1) - np.sqrt(Distribution2)), 2)) / 2

            result[index,index2] = theta*np.power(d0,2) + (1 - theta)*np.power(d1,2)

    return result


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

df = df[['timestamp', 'id', 'y']]

#r = distance(df=df)

col = 'y'
lags = 120
df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(lags).std())
df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_std_'+str(lags)}), on = ['timestamp','id'])

df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(lags).mean())
df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_'+str(lags)}), on = ['timestamp','id'])

timestamps = df.timestamp.unique()
selected_timestamps = timestamps[(( timestamps % lags ) == 0)]
selected_timestamps = np.delete(selected_timestamps, 0)

df2 = df[df.timestamp.isin(selected_timestamps)]
del df2['y']
del df2['y_ma_'+str(lags)]

pivoted = df2.pivot('id', 'timestamp')
pivoted.fillna(0, inplace=True)

db = DBSCAN(eps=0.01, min_samples=3).fit(pivoted)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

#Range for k
kmin = 2
kmax = 60
sil_scores = []

#Compute silouhette scoeres
for k in range(kmin,kmax,2):
    km = KMeans(n_clusters=k, n_init=20).fit(pivoted)
    sil_scores.append(silhouette_score(pivoted, km.labels_))

#Plot
plt.plot(range(kmin,kmax,2), sil_scores)
plt.title('KMeans Results')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_ = 8

km = KMeans(n_clusters=n_clusters_, n_init=20).fit(pivoted)
colors=km.labels_
counter=collections.Counter(colors)
print(counter)

n_iter = 5000
plt.figure(1)
i_plot = 1
for i in [5, 15, 30]:
    model = TSNE(n_components=2, n_iter = n_iter, random_state=0, perplexity = i)
    np.set_printoptions(suppress=True)
    Y = model.fit_transform(pivoted)
    plt.subplot(int(str(31)+str(i_plot)))
    plt.scatter(Y[:, 0], Y[:, 1], c= colors)
    plt.title('t-SNE with perplexity = {}'.format(i))
    i_plot += 1

plt.show()

#for i in [30]:
#    fig = plt.figure(1, figsize=(8, 6))
#    ax = Axes3D(fig, elev=-150, azim=110)
#    model = TSNE(n_components=3, random_state=0, perplexity=i, n_iter=n_iter)
#    np.set_printoptions(suppress=True)
#
#    Y = model.fit_transform(pivoted)
#
#    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2],c=colors, cmap=plt.cm.Paired)
#    ax.set_title("3D T-SNE - Perplexity = {}".format(i))
#    ax.set_xlabel("1st dim")
#    ax.w_xaxis.set_ticklabels([])
#    ax.set_ylabel("2nd dim")
#    ax.w_yaxis.set_ticklabels([])
#    ax.set_zlabel("3rd dim")
#    ax.w_zaxis.set_ticklabels([])
#    plt.show()


t = [list(df[['id']].iloc[colors==id].id) for id in np.unique(colors)]
id = 5
ids = t[id]
df_cluster = df[df.id.isin(ids)][['timestamp','id','y']]
df_cluster_pivoted = df_cluster.pivot('timestamp', 'id')
corr_cumsum = df_cluster_pivoted.corr()
dist = corr_cumsum.as_matrix()
plt.hist(dist.flatten(),bins=100)
plt.title("Distribution of Correlations Between Id's");
g = sns.clustermap(dist,metric="euclidean",method="average")
