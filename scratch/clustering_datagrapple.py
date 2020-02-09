import numpy as np
import scipy
from scipy import stats
from scipy.stats import mstats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

def get_dependence_repr(data):
    return scipy.stats.mstats.rankdata(data,1)


def get_distribution_repr(data):
    N = data.shape[0]
    nbBins = 200
    distribution_repr = np.zeros((N,nbBins))
    for i in range(0,N):
        hist, bin_edges = np.histogram(data[i,], bins=nbBins, range=(-10,10), density=True)
        distribution_repr[i,] = hist*np.diff(bin_edges)
    
    return distribution_repr


def get_gnpr(data,theta):
    dependence_repr = get_dependence_repr(data)
    distribution_repr = get_distribution_repr(data)
    
    N = data.shape[0]
    T = data.shape[1]
    L = dependence_repr.shape[1]+distribution_repr.shape[1]
    
    dependence_dist = np.zeros((N,N))
    distrib_dist = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            dependence_dist[i,j] = np.sum(np.power((dependence_repr[i,] - dependence_repr[j,]),2)) / ((1/(3*T))*(T+1)*(T-1))
            distrib_dist[i,j] = np.sum(np.power((np.sqrt(distribution_repr[i,]) - np.sqrt(distribution_repr[j,])),2)) / 2
    
    dep_dist_max = dependence_dist.max()
    distrib_dist_max = distrib_dist.max()
    
    gnpr = np.zeros((N,L))
    for i in range(0,N):
        gnpr[i,0:T] = dependence_repr[i,]*np.sqrt(theta)*np.sqrt(3*T)/np.sqrt((T+1)*(T-1)*dep_dist_max)
        gnpr[i,T:] = np.sqrt(distribution_repr[i,])*np.sqrt(1-theta)/np.sqrt(2*distrib_dist_max)
    
    return gnpr


def create_random_walks(rho_market,rho_cluster,K,N,T):
    random_walks = np.zeros((N,T+1))
    
    market_factor = np.random.normal(0,1,T)
    
    cluster_factors = np.zeros((K,T))
    for k in range(0,K):
        cluster_factors[k,] = np.random.normal(0,1,T)
    
    idiosync_factors = np.zeros((N,T))
    for n in range(0,N):
        if n%2:
            idiosync_factors[n,] = np.random.normal(0,1,T)
        else:
            idiosync_factors[n,] = np.random.laplace(0,1/np.sqrt(2),T)
    
    increments = np.zeros((N,T))
    cluster_class = 0
    size_class = np.floor(N/K)
    
    for n in range(0,N):
        
        market_compo = np.sqrt(rho_market)*market_factor
        indus_compo = np.sqrt(rho_cluster)*cluster_factors[cluster_class,]
        idiosync_compo = np.sqrt(1-rho_market-rho_cluster)*idiosync_factors[n,]
        increments[n,] = market_compo + indus_compo + idiosync_compo
        
        if n%2:
            random_walks[n,T] = 2*cluster_class
        else:
            random_walks[n,T] = 2*cluster_class+1
            
        if(((n+1)%size_class == 0) and (cluster_class < K-1)):
            cluster_class += 1
            
    for n in range(0,N):
        random_walks[n,0] = 100
        for t in range(1,T):
            random_walks[n,t] = random_walks[n,t-1] + increments[n,t]
    
    return random_walks


def differentiate(r_walk):
    N = r_walk.shape[0]
    T = r_walk.shape[1] - 1
    dS = np.diff(r_walk[0:,:T])
    
    return dS


#input parameters of the random walks model
rho_market = 0.1
rho_cluster = 0.1
K = 3
N = 120
T = 10000
nbDistrib = 2
nbCluster = nbDistrib*K

#transforming raw data to GNPR
random_walks = create_random_walks(rho_market,rho_cluster,K,N,T)
dS = differentiate(random_walks)
gnpr = get_gnpr(dS,theta=0.5)


np.random.seed(42)
kmeans = KMeans(init='k-means++', n_clusters=nbCluster, n_init=50)
kmeans.fit(gnpr)
print("Adjusted Rand Index:", metrics.adjusted_rand_score(random_walks[0:,T], kmeans.labels_))

def get_dependence_cl(finest_class):
    return np.floor(finest_class/2)
def get_distribution_cl(finest_class):
    return finest_class % 2

parameters = ((nbDistrib,0,get_distribution_cl(random_walks[0:,T])), \
              (K,1,get_dependence_cl(random_walks[0:,T])),\
              (nbCluster,0.5,random_walks[0:,T]))

for nb_cluster, theta, benchmark in parameters:
    gnpr = get_gnpr(dS,theta)
    kmeans = KMeans(init='k-means++', n_clusters=nb_cluster, n_init=50)
    kmeans.fit(gnpr)
    print("K-Means++ on GNPR:","theta =",theta,"nb_cluster =",nb_cluster)
    print(metrics.adjusted_rand_score(benchmark, kmeans.labels_))
    
    kmeans = KMeans(init='k-means++', n_clusters=nb_cluster, n_init=50)
    kmeans.fit(dS)
    print("K-Means++ on dS:","nb_cluster =",nb_cluster)
    print(metrics.adjusted_rand_score(benchmark, kmeans.labels_),"\n")
    
colors=kmeans.labels_
counter=collections.Counter(colors)
print(counter)

id = 5
for i in range(0,N):
    if colors[i] == id:
        plt.plot(random_walks[i])

plt.show()

