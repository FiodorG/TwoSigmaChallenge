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
from matplotlib.collections import LineCollection
from utilities import remove_expired_ids

np.random.seed(42)
pd.set_option('display.max_columns', 120)
np.set_printoptions(precision=5, suppress=True)

                                    
#########################################################
def plot_cluster(X, labels, model):
    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver='dense', n_neighbors=6)
    embedding = node_position_model.fit_transform(X.T).T

    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    partial_correlations = model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels, cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w', edgecolor=plt.cm.spectral(label / float(n_labels)), alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(), embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(), embedding[1].max() + .03 * embedding[1].ptp())

    plt.show()


#########################################################
def plot_y_in_cluster(df, group=20):
    rcParams['figure.figsize'] = 10, 5
    for key, grp in df[df.id.isin(map(int,list(groups[str(group)])))].groupby(['id']): 
        plt.plot(grp['timestamp'], np.cumsum(grp['y']), label = "id {0:02d}".format(key))
    plt.legend(loc='best')  
    plt.title('y distribution')
    plt.show()


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

df = df[['timestamp', 'id', 'y']]
df = remove_expired_ids(df)
df = df[df.id<1000]
pivoted = df.pivot('id', 'timestamp')
pivoted.fillna(0, inplace=True)
x = np.array(pivoted)
X = x.copy().T
X /= X.std(axis=0)

#model = covariance.GraphLassoCV(cv=2, n_jobs=-1)
model = covariance.GraphLasso(alpha=1e-1)
model.fit(X)

sparse_covariance = model.covariance_
names = df.id.unique().astype(np.str)
_, labels = cluster.affinity_propagation(sparse_covariance)
n_labels = labels.max()

groups = {}
for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
    groups[str(i)] = names[labels == i].astype(np.int)

plot_y_in_cluster(df, group=3)
plot_cluster(X, labels, model)
