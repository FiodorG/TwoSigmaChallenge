def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
from sklearn import (svm, linear_model, preprocessing, ensemble)
from libraries import kagglegym
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import scipy.signal
import random
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn import (svm, linear_model, preprocessing, ensemble, decomposition, manifold, covariance, cluster)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import PolynomialFeatures
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
from sklearn.model_selection import train_test_split
import timeit
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#import xgboost as xgb
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, is_valid_linkage
from itertools import chain
from pylab import rcParams
from sklearn.metrics import r2_score

np.random.seed(42)
pd.set_option('display.max_columns', 120)
np.set_printoptions(precision=5, suppress=True)


with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")
    
df = df[df.id < 500]
df = df[['timestamp', 'id', 'y']]
df_pivoted = pd.pivot_table(df, values='y', index=['timestamp'], columns=['id'], aggfunc=np.sum)
df_pivoted.fillna(0, inplace=True)
df_pivoted.to_csv('assets.csv',index=False)
df_pivoted=pd.read_csv('assets.csv')

cor=df_pivoted.corr(method='spearman')
cor.loc[:,:] = np.tril(cor, k=-1)
cor = cor.stack()

ones = cor[np.abs(cor) > 0.3].reset_index().loc[:,['level_0','level_1']]
ones = ones.query('level_0 not in level_1')
groups = ones.groupby('level_0').agg(lambda x: set(chain(x.level_0,x.level_1))).values
print('groups of assets which are correlated on y value more then 0.4')
assets_in_groups = []
for g,i in zip(groups,range(len(groups))):
    assets_in_groups = assets_in_groups + list(g[0])
    print(i,g)

assets_in_groups.sort()
print('all assets: ' + str(len(df.id.unique())) )
print('assets in groups: ' + str(len(set(assets_in_groups))) )
    
group = 3
rcParams['figure.figsize'] = 10, 5
for key, grp in df[df.id.isin(map(int,list(groups[group][0])))].groupby(['id']): 
    plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))
plt.legend(loc='best')  
plt.title('y distribution')
plt.show()