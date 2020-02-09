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
from utilities import r_score, df_columns
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from hmmlearn import hmm

np.random.seed(42)


#########################################################
def clean_data_train(df_train, columns):
    df_train = df_train[['id', 'timestamp'] + columns + ['y']]
    df_train = df_train.fillna(0)
    
    if len(df_train.columns) == 4 and 'technical_20' in df_train.columns:
        df_train = df_train[df_train.technical_20 != 0]

    low_y_cut = np.percentile(df_train.y, 5)
    high_y_cut = np.percentile(df_train.y, 95)

    y_is_above_cut = (df_train.y > high_y_cut)
    y_is_below_cut = (df_train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    
    return df_train[y_is_within_cut]


#########################################################
def cluster_ids(dataframe):
    unique_ids = pd.unique(dataframe.id)
    NaN_vectors = np.zeros(shape=(len(unique_ids), dataframe.shape[1]))

    for i, i_id in enumerate(unique_ids):
        data_sub = dataframe[dataframe.id == i_id]
        NaN_vectors[i, :] = np.sum(data_sub.isnull(), axis=0) / float(data_sub.shape[0])

    bin_NaN = 1 * (NaN_vectors == 1)
    bin_cov = np.corrcoef(bin_NaN.T)
    edges = []

    # count i,j is the number of id for which both columns are missing
    count = np.dot(bin_NaN.T, bin_NaN)
    for i in range(bin_cov.shape[0]):
        for j in range(bin_cov.shape[0] - i):
            if i != i+j and bin_cov[i, i+j] >= 0.9:
                edges.append([i, i+j, count[i, i+j]])

    nan_features = np.zeros((bin_NaN.shape[0],len(edges)))
    for i in range(bin_NaN.shape[0]):
        for j, edge in enumerate(edges):
            nan_features[i,j] = 1*(bin_NaN[i,edge[0]] & bin_NaN[i,edge[1]])

    db = DBSCAN(eps=0.3, min_samples=10).fit(nan_features)
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    km = KMeans(n_clusters, n_init=20).fit(nan_features)

    id_clusters = pd.DataFrame(km.labels_, index=np.unique(dataframe.id), columns=['cluster'])
    return id_clusters


#########################################################
def add_features(df, columns):
    for col in columns:
        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_20'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_200'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_20'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_200'}), on = ['timestamp','id'])

    df = df.fillna(method='bfill')

    return df


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    dataframe = data_file.get("train")

#id_clusters = cluster_ids(dataframe)
#frequencies = collections.Counter(id_clusters.cluster)
#print(frequencies)
#cluster = 1
#cluster_ids = np.array(id_clusters[id_clusters.cluster == cluster].index)
#df = dataframe[dataframe.id.isin(cluster_ids)]           

df = dataframe
columns = df_columns(df)

df_train = df[df.timestamp <= 1000]
df_test = df[df.timestamp > 1000]

y = df_test.y

df_train = clean_data_train(df_train, columns)
df_test = df_test[['id', 'timestamp'] + columns]

x_train = df_train[columns]
y_train = df_train[['y']]

model = linear_model.Ridge(normalize=True)
selector = RFECV(model, step=1, cv=2)
selector = selector.fit(x_train, y_train)
selected_columns = [columns[i] for i in np.where(selector.support_==True)[0]]
#print("Optimal number of features : %d" % selector.n_features_)
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
#plt.show()

model = linear_model.LarsCV(max_iter=200, normalize=True, cv=5, n_jobs=-1)
model.fit(x_train[selected_columns], y_train)
selected_columns = [selected_columns[col_id] for col_id in model.active_]
print(selected_columns)

for col in selected_columns:
    model1 = linear_model.LarsCV(max_iter=200, normalize=True, cv=2, n_jobs=-1)
    model1.fit(x_train[[col]], y_train)
    x_test = df_test[[col]].fillna(df_train.mean(axis=0))
    y_test = model1.predict(x_test)
    print(col + ': ' + str(r_score(y, y_test)))

x_test = df_test[selected_columns].fillna(df_train.mean(axis=0))
y_test = model.predict(x_test[selected_columns])

print('global: ' + str(r_score(y, y_test)))

#plt.figure(1)
#plt.plot(df_test.timestamp, y)
#plt.plot(df_test.timestamp, y_fit)
#plt.show()
