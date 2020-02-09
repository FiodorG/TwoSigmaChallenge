import numpy as np
import matplotlib.pyplot as plt
from sklearn import (svm, linear_model, preprocessing, ensemble)
from libraries import kagglegym
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import scipy.signal
import random
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble)
from sklearn.metrics import r2_score


#########################################################
def clean_data_train(df_train, columns):
    df_train = df_train[['id', 'timestamp'] + columns + ['y']]
#    df_train = df_train.fillna(df_train.median(axis=0))
    df_train = df_train.fillna(0)
#    df_train = df_train[df_train.technical_20 != 0]

    low_y_cut = np.percentile(df_train.y, 5)
    high_y_cut = np.percentile(df_train.y, 95)

    y_is_above_cut = (df_train.y > high_y_cut)
    y_is_below_cut = (df_train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    
    return df_train[y_is_within_cut]


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
def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r
    

#########################################################
def get_weighted_y(y_fit, df_train, df_test):
    ymedian_dict = dict(df_train.groupby(["id"])["y"].median())
    
    id = np.array(df_test.id)
    
    y = np.zeros(len(id));
    for i in np.arange(len(id)):
        y[i] = 1. * y_fit[i] + 0.00 * ymedian_dict[id[i]] if id[i] in ymedian_dict else y_fit[i]
        
    return y

#########################################################
def df_columns(df):    
    columns_derived = [c for c in df.columns if 'derived' in c]
    columns_fundamental = [c for c in df.columns if 'fundamental' in c]
    columns_technical = [c for c in df.columns if 'technical' in c]
    all_columns = columns_derived + columns_fundamental + columns_technical
    return all_columns
    
    
#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")   
    
#df = df.fillna(df.median(axis=0))
    
#
#columns = ['technical_20']
#for col in columns:
#    df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).mean())
#    df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_20'}), on = ['timestamp','id'])

#columns = ['technical_20', 'technical_20_ma_20']
#columns = ['technical_40']
    
#id_clusters = cluster_ids(df_train)

selected_columns = []
best_scores = []
for id in np.unique(df.id):

    df_temp = df[df.id==id]
    columns = df_columns(df_temp)
    df_train = df_temp[df_temp.timestamp <= 1000]  

    if df_train.empty:
        continue
  
    df_test = df_temp[df_temp.timestamp > 1000]   
    y = df_test.y

    df_train = clean_data_train(df_train, columns)
    df_train.dropna(axis=1, inplace=True, how='all')
    columns = df_columns(df_train)
    df_test = df_test[['id', 'timestamp'] + columns]

    if df_test.empty:
        continue
    
    model = linear_model.LarsCV(max_iter=200, normalize=True, cv=2, n_jobs=-1)
    #model = linear_model.ElasticNetCV()
    #model = linear_model.BayesianRidge()

    x_train = df_train[columns]
    y_train = df_train[['y']]

    model.fit(x_train, y_train)
    selected_columns.append([columns[col_id] for col_id in model.active_])

    x_test = df_test[columns].fillna(0)
    y_test = model.predict(x_test)
    #y_fit = get_weighted_y(y_test, df_train, df_test)

    best_scores.append(r_score(y, y_test))

# results are due to selection in sample!!!
print(np.average(best_scores))    
    
plt.figure(1)
plt.bar(np.arange(len(best_scores)), best_scores)
plt.show()

#plt.figure(1)
#plt.plot(df_test.timestamp, y)
#plt.plot(df_test.timestamp, y_fit)
#plt.show(

