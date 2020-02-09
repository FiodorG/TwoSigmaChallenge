import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libraries import visualization
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from time import time

pd.set_option('display.max_columns', 120)

with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")
    
unique_ids = pd.unique(df.id)
NaN_vectors = np.zeros(shape=(len(unique_ids), df.shape[1]))

for i, i_id in enumerate(unique_ids):
    data_sub = df[df.id ==i_id]
    NaN_vectors[i,:] = np.sum(data_sub.isnull(),axis=0) /float(data_sub.shape[0])
    
plt.matshow(np.transpose(NaN_vectors))
plt.show()

bin_NaN = 1 * (NaN_vectors == 1)
plt.matshow(np.transpose(bin_NaN))
plt.show()

bin_cov=np.corrcoef(bin_NaN.T)
bin_cov.shape[1]
plt.matshow(bin_cov)
plt.show()

plt.matshow(bin_cov >= 0.9) 
plt.show()

edges = []

# count i,j is the number of id for which both columns are missing
count = np.dot(bin_NaN.T,bin_NaN)
for i in range(bin_cov.shape[0]):
    for j in range(bin_cov.shape[0]-i):
        if i != i+j and bin_cov[i,i+j] >= 0.9:
            edges.append([i,i+j, count[i,i+j]])
print(edges)

ucount = [i[2] for i in edges]
print(np.unique(ucount))

nan_features = np.zeros((bin_NaN.shape[0],len(edges)))
for i in range(bin_NaN.shape[0]):
    for j, edge in enumerate(edges):
        nan_features[i,j] = 1*(bin_NaN[i,edge[0]] & bin_NaN[i,edge[1]])
        
###############################################################################
X = nan_features

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

k=n_clusters_
km = KMeans(n_clusters=k, n_init=20).fit(nan_features)
colors=km.labels_

n_iter = 5000

for i in [15]:
    t0 = time()
    model = TSNE(n_components=2, n_iter = n_iter, random_state=0, perplexity = i)
    np.set_printoptions(suppress=True)
    Y = model.fit_transform(nan_features)
    t1 = time()

    print( "t-SNE: %.2g sec" % (t1 -t0))
    plt.scatter(Y[:, 0], Y[:, 1], c= colors)
    plt.title('t-SNE with perplexity = {}'.format(i))
    plt.show()