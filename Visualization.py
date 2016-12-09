from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.tools.plotting as pdtools
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def plot_column(dataset, column, id, cumsum=False, index_start=0, index_end=-1):
    dataset = dataset[dataset['id'] == id]

    if index_end != -1:
        values = np.cumsum(dataset[column][index_start:index_end])
        times = np.arange(index_end - index_start)
    else:
        values = dataset[column][index_start:]
        times = np.arange(len(values) - index_start)

    if cumsum:
        values = np.cumsum(values)

    plt.plot(times, values)
    plt.xlabel(column)
    axes = plt.gca()
    axes.set_xlim([0, times[-1]])
    plt.show()


def show_y_stats(dataset):
    n = dataset['timestamp']
    y_mean = dataset['y']['mean']
    y_std = dataset['y']['std']

    plt.plot(n, y_mean, '.')
    plt.xlabel('n')
    plt.ylabel('$y_{n}^{Mean}$')
    plt.show()
    plt.savefig('yMean.png')

    plt.plot(n, y_std, '.')
    plt.xlabel('n')
    plt.ylabel('$y_{n}^{Std}$')
    plt.show()
    plt.savefig('yStd.png')


def plot_all_graphs(df):
    i = 0
    n = 5
    columns = df.columns
    while True:
        if i >= len(columns):
            break
        for j in range(n):
            if i >= len(columns):
                break

            if j == 0:
                plt.figure(figsize=(8, 8.0 / n))

            plt.axes([j * 1.0 / n, 0, 1.0 / n, 1])
            plt.plot(df['timestamp'], df[columns[i]])
            plt.xticks([])
            plt.yticks([])
            i += 1
    plt.show()
    
    
def plot_column_per_id_and_timestamp(dataset, column):
    dataset = dataset[['id', 'timestamp', column]]
    instr_time = dataset.groupby('id').agg({'timestamp': [np.min, np.max]})
    sort_order = instr_time.sort_values(by=[('timestamp', 'amin'), ('timestamp', 'amax')]).index
    # df[column] += 0.2
    df1 = dataset.set_index(['id', 'timestamp']).unstack('id').reindex(sort_order)
    np_array = df1.fillna(0).values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np_array, cmap='gray')
    plt.show()


def plot_autocorrelation(dataset, column, id):
    dataset = dataset[['id', 'timestamp', column]]
    pdtools.autocorrelation_plot(dataset[dataset.id == id])


def plot_pca(dataset, column):
    dataset = dataset[['id', 'timestamp', column]]
    df = dataset.set_index(['id', 'timestamp']).unstack('id').fillna(0)
    dim = df.shape
    # print(df.head())
    # print(dim)
    n = 200
    pca = PCA(n_components=n, whiten=True)
    pca.fit(df)

    plt.figure(1)
    plt.subplot(511)
    plt.bar(np.arange(n), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
    plt.step(np.arange(n), np.cumsum(pca.explained_variance_ratio_), where='mid', label='cumulative explained variance')
    plt.xlabel('Number of chosen ids')
    plt.ylabel('Cumulative explained variance')
    plt.xlim([0, n])

    pca2 = PCA(n_components=2, whiten=True)
    reduced_data = pca2.fit_transform(df)

    plt.subplot(512)
    plt.plot(reduced_data[:, 0], 'r')
    plt.ylabel('PC1')
    plt.xlabel('Timestamp')
    plt.xlim([0, dim[0]])

    plt.subplot(513)
    plt.plot(reduced_data[:, 1], 'g')
    plt.ylabel('PC2')
    plt.xlabel('Timestamp')
    plt.xlim([0, dim[0]])

    plt.subplot(514)
    plt.bar(np.arange(dim[1]), pca.components_[0, :])
    plt.ylabel('Weight in PC1')
    plt.xlabel('Id')
    plt.xlim([0, dim[1]])

    plt.subplot(515)
    plt.bar(np.arange(dim[1]), pca.components_[1, :])
    plt.ylabel('Weight in PC2')
    plt.xlabel('Id')
    plt.xlim([0, dim[1]])

    plt.show()


def plot_stock_prices(dataset, column, n, cumsum):
    for i, (idVal, dfG) in enumerate(dataset[['id', 'timestamp', column]].groupby('id')):
        if i > n:
            break
        df1 = dfG[['timestamp', column]].groupby('timestamp').agg(np.mean).reset_index()

        if cumsum:
            series = np.cumsum(df1[column])
        else:
            series = df1[column]

        plt.plot(df1['timestamp'], series, label='%d' % idVal)
        plt.xlim([0, 1824])


def plot_all_columns_for_id(dataset, id, columns_derived, columns_fundamental, columns_technical):
    dataset = dataset[dataset.id == id].fillna(0)
    df_y = dataset[['y']]
    df_derived = dataset[columns_derived].drop(['id', 'timestamp'], axis=1)
    df_fundamental = dataset[columns_fundamental].drop(['id', 'timestamp'], axis=1)
    df_technical = dataset[columns_technical].drop(['id', 'timestamp'], axis=1)

    plt.figure(1)
    plt.subplot(411)
    plt.title('id: %d' % id)
    plt.plot(dataset['timestamp'], df_y)
    plt.ylabel('y')
    plt.subplot(412)
    plt.plot(dataset['timestamp'], df_derived)
    plt.ylabel('derived')
    plt.subplot(413)
    plt.plot(dataset['timestamp'], df_fundamental)
    plt.ylabel('fundamental')
    plt.subplot(414)
    plt.plot(dataset['timestamp'], df_technical)
    plt.ylabel('technical')
    plt.xlabel('timestamp')
    plt.show()


def linear_regression(dataset, id, column_to_predict, n):
    dataset = dataset[dataset.id == id]
    x = dataset.drop(['id', 'timestamp', column_to_predict], axis=1).fillna(0).values
    y = dataset[[column_to_predict]].fillna(0).values[:, 0]

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

