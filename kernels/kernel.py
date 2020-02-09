import numpy as np
import matplotlib.pyplot as plt
from sklearn import (svm, linear_model, preprocessing, ensemble)
from libraries import kagglegym
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import scipy.signal
import random

random.seed(999)

# pd.set_option('display.max_columns', 120)
# pd.options.mode.chained_assignment = None
columns = ['technical_20']
#columns = ['technical_20', 'fundamental_53', 'technical_30', 'technical_27', 'derived_0']


#########################################################
def clean_data_train(df_train):
    df_train = df_train[['id', 'timestamp'] + columns + ['y']]
    df_train = df_train.fillna(df_train.median(axis=0))
#    df_train = df_train[df_train.technical_20 != 0]

    low_y_cut = np.percentile(df_train.y, 5)
    high_y_cut = np.percentile(df_train.y, 95)

    y_is_above_cut = (df_train.y > high_y_cut)
    y_is_below_cut = (df_train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

    df_train_inliers = df_train.loc[y_is_within_cut]
    df_train_outliers = df_train.loc[~y_is_within_cut]

    scaler_x = preprocessing.StandardScaler().fit(df_train_inliers[columns])
    df_train_inliers.loc[:, columns] = scaler_x.transform(df_train_inliers[columns])

    scaler_y = preprocessing.StandardScaler().fit(df_train_inliers['y'].reshape(-1, 1))
    df_train_inliers.loc[:, ['y']] = scaler_y.transform(df_train_inliers['y'].reshape(-1, 1))

    return df_train_inliers, df_train_outliers, scaler_x, scaler_y


#########################################################
def cluster_ids(df_train):
    unique_ids = pd.unique(df_train.id)
    NaN_vectors = np.zeros(shape=(len(unique_ids), df_train.shape[1]))

    for i, i_id in enumerate(unique_ids):
        data_sub = df_train[df_train.id == i_id]
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

    id_clusters = pd.DataFrame(km.labels_, index=np.unique(df_train.id), columns=['cluster'])
    return id_clusters


#########################################################
def add_features(df):

#    for col in columns:
#        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).mean())
#        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_20'}), on = ['timestamp','id'])
#
#        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).mean())
#        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_60'}), on = ['timestamp','id'])
#
#        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).mean())
#        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_200'}), on = ['timestamp','id'])
#
#    df = df.fillna(method='bfill')

    return df


#########################################################
def train_models_inliers(df, id_clusters):
    models = {}

    clusters = np.unique(id_clusters)

    for current_cluster in clusters:
        current_cluster_index = id_clusters[id_clusters == current_cluster]
        current_cluster_ids = np.array(current_cluster_index.dropna().index)
        df_current_cluster = df[df.id.isin(current_cluster_ids)]
        models[str(current_cluster)] = \
            train_models(df_current_cluster.drop(['id', 'timestamp', 'y'], axis=1), df_current_cluster.y)

    models['global'] = train_models(df.drop(['id', 'timestamp', 'y'], axis=1), df.y)

    return models


#########################################################
def train_models(x, y):
    model1 = linear_model.Lars(n_nonzero_coefs=1)
    model2 = linear_model.ElasticNetCV()
    model3 = linear_model.BayesianRidge()
    model1.fit(x, y)
    model2.fit(x, y)
    model3.fit(x, y)
    return [model1, model2, model3]


#########################################################
def train_models_outliers(df, id_clusters):
    y_train = df.y
    x_train = df.drop(['y'], axis=1)

    models = {}
    return models


#########################################################
def clean_data_test(df_train, df_test, scaler_x):
    df_test = df_test[['id', 'timestamp'] + columns]
    df_test = df_test.fillna(df_train[['id', 'timestamp'] + columns].mean())
    df_test.loc[:, columns] = scaler_x.transform(df_test[columns])

    timestamp = df_test["timestamp"][1]
    return df_test, timestamp


#########################################################
def append_observation_to_test_df(df_test, x_test):
    df = df_test[['id', 'timestamp'] + columns]
    df = df.append(x_test, ignore_index=True)
    return df


#########################################################
def inverse_transform(y_test, scaler_y):
    return scaler_y.inverse_transform(y_test)


#########################################################
def predict(models, df_test, id_clusters, rewards):

    models_inliers = models['models_inliers']
    models_outliers = models['models_outliers']

    df_test = df_test[['id'] + columns]
    y_predict = pd.DataFrame({'id': df_test.id, 'y': np.zeros(len(df_test))})

    for current_cluster in np.unique(id_clusters.cluster):
        current_cluster_ids = np.array(id_clusters[id_clusters.cluster == current_cluster].index)
        models_inliers_for_id = models_inliers[str(current_cluster)]
        x_test = df_test[df_test.id.isin(current_cluster_ids)]
        y_predict_for_id = [model.predict(x_test.drop(['id'], axis=1)) for model in models_inliers_for_id]
        y_predict.loc[y_predict.id.isin(current_cluster_ids), ['y']] = (np.median(y_predict_for_id, axis=0))

    models_inliers_global = models_inliers['global']
    x_test = df_test[y_predict.y == 0]

    if not x_test.empty:
        y_predict_for_id = [model.predict(x_test.drop(['id'], axis=1)) for model in models_inliers_global]
        y_predict.loc[y_predict.y == 0, ['y']] = (np.median(y_predict_for_id, axis=0))

    return y_predict


#########################################################
#########################################################
env = kagglegym.make()
observation = env.reset()

df_train = observation.train
id_clusters = cluster_ids(df_train)
df_inliers, df_outliers, scaler_x, scaler_y = clean_data_train(df_train)
df_inliers = add_features(df_inliers)

models_inliers = train_models_inliers(df_inliers, id_clusters)
models_outliers = train_models_outliers(df_inliers, id_clusters)

models = {
    'models_inliers': models_inliers,
    'models_outliers': models_outliers,
}

#done = False
#n = 0
#arr = []
#while not done:
#    arr.append(observation.features)
#    observation, _, done, _ = env.step(observation.target)
#    print(n)
#    n += 1
#
#df = pd.DataFrame(arr[0])
#for line in arr[1:]:
#    df.append(line, ignore_index=True)

done = False
n = 0
rewards = []
while not done:

    df_test, timestamp = clean_data_test(df_inliers, observation.features, scaler_x)

    y_test = predict(models, df_test, id_clusters, rewards)
    observation.target.y = inverse_transform(y_test, scaler_y)

    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(observation.target)

    if done:
        print("Public score: {}".format(info["public_score"]))

    rewards.append(reward)
    n += 1
