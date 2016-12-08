from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.tools.plotting as pdtools
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA

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
