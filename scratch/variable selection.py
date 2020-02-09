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
from libraries import utilities
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble, decomposition)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
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
def add_features(df, columns):
    for col in columns:
        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_20'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).mean())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_ma_200'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(20).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_vol_20'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(60).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_vol_60'}), on = ['timestamp','id'])

        df_ma = (df.set_index('timestamp').groupby('id')[col].rolling(200).std())
        df = pd.merge(df, df_ma.reset_index().rename(columns = {col: col+'_vol_200'}), on = ['timestamp','id'])

    df = df.fillna(method='bfill')

    return df


#########################################################
with pd.HDFStore("input/train.h5", "r") as data_file:
    dataframe = data_file.get("train")       

df = dataframe[dataframe.id == 500]
columns = utilities.df_columns(df)
#df = add_features(df, columns)
#columns = utilities.df_columns(df)
#columns = ['technical_33']

df_train = df[df.timestamp <= 1000]
df_test = df[df.timestamp > 1000]

y = df_test.y

df_train = clean_data_train(df_train, columns)
df_test = df_test[['id', 'timestamp'] + columns]

x_train = df_train[columns]
y_train = df_train[['y']]

#kbest = SelectKBest(f_regression)
#pipeline = Pipeline([('kbest', kbest), ('lr', linear_model.LinearRegression)])
#grid_search = GridSearchCV(pipeline, {'kbest__k': [1]}, cv=2)

#pipeline = Pipeline([('lr', linear_model.Lars(normalize=True))])
#grid_search = GridSearchCV(pipeline, {'lr__n_nonzero_coefs': [1,2,3,4,5]}, n_jobs=-1)
#grid_search.fit(x_train, y_train)
#t = grid_search.best_estimator_
#selected_columns = [columns[col_id] for col_id in t.named_steps['lr'].active_]
#print(grid_search.best_params_)
#print(grid_search.best_score_)

#model = linear_model.Ridge(alpha=50, normalize=True)
#selector = RFECV(model, step=1, cv=5)
#selector = selector.fit(x_train, y_train)
#selected_columns = [columns[i] for i in np.where(selector.support_==True)[0]]

tscv = TimeSeriesSplit(n_splits=2)

cv_splits = []
for train_index, test_index in tscv.split(x_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    cv_splits.append([list(train_index), list(test_index)])

pca = decomposition.PCA(whiten=True)
model = linear_model.LinearRegression(normalize=True)
pipe = Pipeline(steps=[('pca', pca), ('lr', model)])

#pca.fit(x_train)
#plt.figure(1, figsize=(4, 3))
#plt.clf()
#plt.axes([.2, .2, .7, .7])
#plt.plot(pca.explained_variance_, linewidth=2)
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')

n_components = np.arange(1,21)
estimator = GridSearchCV(pipe, dict(pca__n_components=n_components), cv=cv_splits)
estimator.fit(x_train, y_train)
n_components = estimator.best_estimator_.named_steps['pca'].n_components
print(n_components)
                                                    
pca = decomposition.PCA(whiten=True, n_components=3)
model = linear_model.LinearRegression(normalize=True)
pipe = Pipeline(steps=[('pca', pca), ('lr', model)])                          
pipe.fit(x_train, y_train)   
y_test = pipe.predict(x_test[columns])                          
print('global: ' + str(utilities.r_score(y, y_test)))

model = linear_model.LarsCV(max_iter=1000, normalize=True, cv=cv_splits, n_jobs=-1)
#model = linear_model.LassoCV(max_iter=1000, normalize=True, cv=cv, n_jobs=-1)
model.fit(x_train, y_train)


N = x_train.shape[0]
splits = 10
idxs = np.arange(N)
cv_splits = [(idxs[:i], idxs[i:]) for i in range(int(N/splits)+1, N, int(N/splits))]


rfecv = RFECV(estimator=linear_model.Ridge(normalize=True), step=1, cv=cv_splits)
rfecv.fit(x_train, y_train)
selected_columns = [columns[i] for i in np.where(rfecv.support_==True)[0]]
print(selected_columns)

model = linear_model.LarsCV(max_iter=1000, normalize=True, cv=cv_splits, n_jobs=-1)
model.fit(x_train, y_train)

#train_sizes, train_scores, valid_scores = learning_curve(model, x_train, y_train, cv=2)
#train_scores, valid_scores = validation_curve(linear_model.Ridge(), x_train, y_train, "alpha", np.logspace(-7, 3, 3))

model = linear_model.LarsCV(max_iter=200, normalize=True, cv=2, n_jobs=-1)
model.fit(x_train, y_train)

selected_columns = [columns[col_id] for col_id in model.active_]
print(selected_columns)

score = []
for col in columns:
#    model1 = linear_model.LarsCV(max_iter=200, normalize=True, cv=2, n_jobs=-1)
    model1 = linear_model.LinearRegression(normalize=True)
    model1.fit(x_train[[col]], y_train)
    x_test = df_test[[col]].fillna(df_train.mean(axis=0))
    y_test = model1.predict(x_test)
    col_score = utilities.r_score(y, y_test)
    score.append(col_score)
    if col_score > 0:
        print(col + ': ' + str(col_score))
        
score = []
for col in columns:
#    model1 = linear_model.LarsCV(max_iter=200, normalize=True, cv=2, n_jobs=-1)
    model1 = linear_model.LinearRegression(normalize=True)
    model1.fit(np.diff(np.array(x_train[[col]]), axis=0), np.diff(np.array(y_train), axis=0))
    x_test = df_test[[col]].fillna(df_train.mean(axis=0))
    y_test = model1.predict(np.diff(np.array(x_test), axis=0))
    col_score = utilities.r_score(np.diff(np.array(y)), y_test)
    score.append(col_score)
    if col_score > 0:
        print(col + ': ' + str(col_score))

x_test = df_test[columns].fillna(df_train.mean(axis=0))
y_test = model.predict(x_test[columns])
print('global: ' + str(utilities.r_score(y, y_test)))

plt.figure(1)
plt.bar(np.arange(len(columns)), score)
#plt.plot(df_test.timestamp, y)
#plt.plot(df_test.timestamp, y_fit)
plt.show()
