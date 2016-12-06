from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import autocorrelation_plot


def plot_column(dataset, column, index_start=0, index_end=-1):

    if index_end != -1:
        values = np.cumsum(dataset[column][index_start:index_end])
        times = np.arange(index_end - index_start)
    else:
        values = dataset[column][index_start:]
        times = np.arange(len(values) - index_start)

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


with pd.HDFStore("train.h5", "r") as data_file:
    dataset = data_file.get("train")

column_all = dataset.columns
columns_derived = [c for c in dataset.columns if 'derived' in c] + ['timestamp']
columns_fundamental = [c for c in dataset.columns if 'fundamental' in c] + ['timestamp']
columns_technical = [c for c in dataset.columns if 'technical' in c] + ['timestamp']

#plot_column(dataset, 'id', 0, 1000)

df = dataset[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
df_derived = dataset[columns_derived].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
#df_fundamental = dataset[columns_fundamental].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
#df_technical = dataset[columns_technical].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()

df2 = dataset[['id', 'timestamp', 'y']]
temp = df2.groupby('id').apply(len).sort_values(ascending=False).reset_index()
print(temp.head(5))

temp2 = df2[df2['id'].isin(temp['id'][1:2].values)]
temp2 = temp2[['timestamp', 'y']]
print(temp2)
plt.figure(figsize=(8, 4))
plt.plot(temp2['timestamp'], np.cumsum(temp2['y']))
plt.xlabel('timestamp')
plt.ylabel('y')
plt.show()

#yMean = df1['y']['mean']
#yMean = dataset['y'].head(n=1000)
#plt.figure()
#autocorrelation_plot(yMean)

#plot_all_graphs(df_derived)
#yMean = df['y']['mean']
#corrColumns = [c for c in df_derived.columns if 'timestamp' not in c]
#derived = np.array([df_derived[c] for c in corrColumns])
#Corr = np.corrcoef(yMean, derived)
#sns.clustermap(pd.DataFrame(np.abs(Corr), columns=['---y--']+[c[0].split('_')[1] for c in corrColumns]))

#plt.show()
print()
