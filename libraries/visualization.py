from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pandas.tools.plotting as pdtools
from sklearn import linear_model
from sklearn.decomposition import PCA
    
    
#########################################################
def plot_column_all_ids_all_timestamps(df, column):
    df = df[['id', 'timestamp', column]]
    max_timestamp = df.groupby('id').agg({'timestamp': [np.max]})
    sort_order = max_timestamp.sort_values(by=[('timestamp', 'amax')]).index
    # df[column] += 0.2
    df1 = df.set_index(['id', 'timestamp']).unstack('id').reindex(sort_order)
    np_array = df1.fillna(0).values

    fig, ax = plt.subplots()
    ax.imshow(np_array, cmap='gray')
    plt.show()


#########################################################
def plot_columns_many(df, column, n, cumsum):
    for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', column]].groupby('id')):
        if i > n:
            break
        df1 = dfG[['timestamp', column]].groupby('timestamp').agg(np.mean).reset_index()

        if cumsum:
            series = np.cumsum(df1[column])
        else:
            series = df1[column]

        plt.plot(df1['timestamp'], series, label='%d' % idVal)
        plt.xlim([0, 1824])
    plt.show()


#########################################################
def plot_all_columns_for_id_aggregated(df, id, columns_columns_derived, columns_columns_fundamental, columns_technical):
    df = df[df.id == id].fillna(0)
    df_y = df[['y']]
    df_columns_derived = df[columns_columns_derived].drop(['id', 'timestamp'], axis=1)
    df_columns_fundamental = df[columns_columns_fundamental].drop(['id', 'timestamp'], axis=1)
    df_technical = df[columns_technical].drop(['id', 'timestamp'], axis=1)

    plt.figure(1)
    plt.subplot(411)
    plt.title('id: %d' % id)
    plt.plot(df['timestamp'], df_y)
    plt.ylabel('y')
    plt.subplot(412)
    plt.plot(df['timestamp'], df_columns_derived)
    plt.ylabel('columns_derived')
    plt.subplot(413)
    plt.plot(df['timestamp'], df_columns_fundamental)
    plt.ylabel('columns_fundamental')
    plt.subplot(414)
    plt.plot(df['timestamp'], df_technical)
    plt.ylabel('technical')
    plt.xlabel('timestamp')
    plt.show()


#########################################################
def linear_regression(df, id, column_to_predict, n):
    df = df[df.id == id]
    x = df.drop(['id', 'timestamp', column_to_predict], axis=1).fillna(0).values
    y = df[[column_to_predict]].fillna(0).values[:, 0]

    pca = PCA(n_components=n, whiten=True)
    x = pca.fit_transform(x)

    alphas_lars, _, coef_path_lars = linear_model.lars_path(x, y, method='lars')
    # coef_path_cont_lars = interpolate.interp1d(alphas_lars[::-1], coef_path_lars[:, ::-1])
    xx = np.sum(np.abs(coef_path_lars.T), axis=1)
    xx /= xx[-1]
    plt.plot(xx, coef_path_lars.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef|/max|coef|')
    plt.ylabel('Coefficients')
    plt.axis('tight')
    plt.show()

    res = linear_model.Lars(n_nonzero_coefs=n, fit_intercept=True, normalize=True)
    res.fit(x, y)
    print(res.score(x, y))


#########################################################
def plot_all_columns_for_id_detail(df, id, columns=[]):
    
    if columns == []:
        columns_derived = [c for c in df.columns if 'derived' in c]
        columns_fundamental = [c for c in df.columns if 'fundamental' in c]
        columns_technical = [c for c in df.columns if 'technical' in c]
        features_list = columns_derived + columns_fundamental + columns_technical    
    else:
        features_list = columns

    dx = 4
    dy = 5
    fig = plt.figure(figsize=(dy, dx))
    pt_id = df[["timestamp", "y"]][df["id"] == id].dropna()
    fig_num = 1
    for col_str in features_list:
        ax = fig.add_subplot(dy, dx, fig_num)
        pt_id = df[["timestamp", col_str, 'y']][df["id"] == id].dropna()
        if col_str in columns_derived:
            color = "red"
        elif col_str in columns_fundamental:
            color = "blue"
        else:
            color = "green"
        color = "blue"
        ax.scatter(pt_id["timestamp"], pt_id[col_str], s=1, c=color, marker=(1, 2, 0))
#        ax.scatter(pt_id["timestamp"], np.cumsum(pt_id['y']), s=1, c='red', marker=(1, 2, 0))
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xlim(0, 1812)
        ax.get_xaxis().set_visible(False)
        ax.tick_params(bottom="off", top="off", left="off", right="off")
        ax.set_title(col_str, size=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        fig_num += 1

    plt.show()


#########################################################
def histogram_all_columns_for_id_detail(df, id, columns_derived, columns_fundamental, columns_technical):
    df = df[df["id"] == id]
    percentile = 5
    features_list = columns_derived + columns_fundamental + columns_technical
    fig = plt.figure(figsize=(9, 44))
    fig.subplots_adjust(hspace=.5)
    fig_num = 1

    for col_str in features_list:
        nona_column = df[col_str].dropna()
        nona_column = nona_column.sort_values()
        lower_p = math.floor(len(nona_column) * percentile / 100)
        higher_p = math.floor(len(nona_column) * (100 - percentile) / 100)

        ax = fig.add_subplot(15, 8, fig_num)
        if col_str in columns_derived:
            color = "red"
        elif col_str in columns_fundamental:
            color = "blue"
        else:
            color = "green"
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        ax.tick_params(bottom="off", top="off", left="off", right="off")
        ax.set_title(col_str, size=10)
        nona_column[lower_p:higher_p].hist(bins=10, ax=ax, grid=False, color=color)
        fig_num += 1
    plt.show()


#########################################################
def check_random_nan(df):
    features = [x for x in df.columns.values if x not in ['id', 'y', 'timestamp']]

    def make_binary_nan(x):
        if x == x:
            return 0.
        else:
            return 1.

    dfnan = df.applymap(make_binary_nan)
    dfnan['id'] = df['id']
    dfnan['timestamp'] = df['timestamp']
    ids = dfnan['id'].unique()

    df_agg_signal = pd.DataFrame()
    for id in ids:
        df_signal = dfnan[dfnan['id'] == id][features].apply(lambda x: abs(x - x.shift()))
        df_signal.dropna(inplace=True)
        df_agg_signal[id] = df_signal.sum()

    listofIdwithMiddleNaN = []

    for id in df_agg_signal.columns:
        if ([True] in (df_agg_signal[id].values > 1.)):
            listofIdwithMiddleNaN.append(id)

    len(listofIdwithMiddleNaN)


#########################################################
def plot_columns_many_subwindow_ids(df, column, ids, cumsum=False):
    df = df[['id', 'timestamp', column]]
    df = df[df['id'].isin(ids)]

    for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', column]].groupby('id')):
        df1 = dfG[['timestamp', column]].groupby('timestamp').agg(np.mean).reset_index()
        if cumsum:
            series = np.cumsum(df1[column])
        else:
            series = df1[column]

        plt.plot(df1['timestamp'], series, label='id %d' % idVal)
        plt.xlim([0, 1824])

    plt.show()
