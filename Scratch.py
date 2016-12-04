from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
from pandas import read_hdf
import collections
import matplotlib.pyplot as plt

def plot_column(dataset, column, index_start = 0, index_end = -1):

    if index_end != -1:
        values = dataset[column][index_start:index_end]
        times = np.arange(index_end - index_start)
    else:
        values = dataset[column][index_start:]
        times = np.arange(len(values) - index_start)

    plt.plot(times, values)
    plt.xlabel(column)
    axes = plt.gca()
    axes.set_xlim([0, times[-1]])
    plt.show()


dataset = read_hdf('train.h5')
keys = list(dataset.keys())
print(keys)

plot_column(dataset, 'technical_44', 0, 1000)


#counter = collections.Counter(time)
#time = hdf['timestamp']
#keys = list(counter.keys())
#values = list(counter.values())

#plt.hist(values, bins='auto')
#plt.show()

#plt.bar(keys, values)
#plt.ylabel('# data in timestamp')
#plt.xlabel('timestamp')
#plt.show()

print()