import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble)
import scipy.signal

with pd.HDFStore("input/train.h5", "r") as data_file:
    df_train = data_file.get("train")

df_train.fillna(df_train.mean(axis=0), inplace=True)

columns_derived = [c for c in df_train.columns if 'derived' in c]
columns_fundamental = [c for c in df_train.columns if 'fundamental' in c]
columns_technical = [c for c in df_train.columns if 'technical' in c]
columns_all = columns_derived + columns_fundamental + columns_technical

y = df_train[df_train.id==600][['timestamp','y']]
y_filter = scipy.signal.savgol_filter(np.cumsum(y.y), 51, 3)

plt.figure(1)
plt.plot(y.timestamp, np.cumsum(y.y))
plt.plot(y.timestamp, y_filter)
plt.show()

y = df_train[df_train.id==600][['timestamp','y']]
y_filter = scipy.signal.savgol_filter(y.y, 5, 3)

plt.figure(2)
plt.plot(np.array(y.timestamp), np.array(y.y))
plt.plot(np.array(y.timestamp), y_filter)
plt.show()

#y.loc[:, ['y']] = y_filter
#
#plt.figure(1)
##plt.plot(df_train[df_train.id==0].timestamp, np.cumsum(y))
##plt.plot(df_train[df_train.id==0].timestamp, np.cumsum(t))
#plt.plot(df_train[df_train.id==0].timestamp, y)
#plt.plot(df_train[df_train.id==0].timestamp, t)
#plt.show()
#
#plt.figure(1)
#plt.hist(t)
#plt.show()


#    for current_id in np.unique(df_train.id):
#        print(current_id)
#        y = df_train[df_train.id == current_id].y
#        y_filter = scipy.signal.savgol_filter(y, 5, 3)
##        t = np.concatenate([np.array([y.iloc[0]]), np.diff(y_filter)])
#        df_train.loc[df_train.id==current_id, ['y']] = y_filter