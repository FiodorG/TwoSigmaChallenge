from math import *
import numpy as np
import sys
from scipy.spatial.distance import euclidean


x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]], dtype='float')
y = np.array([[2,2], [3,3], [4,4]], dtype='float')

y = np.array([1, 2, 3, 4, 5], dtype='float')
x = np.array([2, 3, 4], dtype='float')
#distance, path = fastdtw(x, y, dist=euclidean)
#best_match, min_distance = find_closest_series(x, y, dist=euclidean)
#print(distance)


#########################################################
with pd.HDFStore("../input/train.h5", "r") as data_file:
    df = data_file.get("train")

id = 500
df = df[df.id == id]
df['y'] = np.cumsum(df.y)

x = scipy.signal.savgol_filter(df[(df.timestamp >= 1000) * (df.timestamp < 1100)].y.values, 11, 3)
y = scipy.signal.savgol_filter(df[(df.timestamp >= 0) * (df.timestamp < 1500)].y.values, 11, 3)

best_match, min_distance, best_indices = find_closest_series(x, y, dist=euclidean)

plt.figure(1)
plt.plot(x, color='red')
#plt.plot(df[(df.timestamp < 1000)].y.values, color='green')
plt.plot(best_match, color='blue')
plt.show()

def find_closest_series(x, y, dist=euclidean):
    if len(x) > len(y):
        raise ValueError('x is longer than y')
    elif len(x) == len(y):
        return y, 0
    else:
        best_indices = []
        min_distance = sys.maxsize
        best_match = y[0:len(x)]
        for i in np.arange(len(y) - len(x)):
#            distance, _ = fastdtw(x, y[i:i + len(x)], dist=euclidean)
            distance = DTWDistance(x, y[i:i + len(x)], 1)
            if distance < min_distance:
                min_distance = distance
                best_match = y[i:i + len(x)]
                best_indices = range(i, i + len(x))
    
        return best_match, min_distance, best_indices

    
def DTWDistance(s1, s2, w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])


def DTW(A, B, window = sys.maxsize, d = lambda x,y: abs(x-y)):
    # create the cost matrix
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = sys.maxsize * np.ones((M, N))

    # initialize the first row and column
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(A[0], B[j])
    # fill in the rest of the matrix
    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    # find the optimal path
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
    
    path.append((0,0))
    return cost[-1, -1], path

def main():
    A, B = np.array([1,2,3,4,2,3]), np.array([1,1,3,3,4,3,3])
    C = np.array([7,8,5,9,11,9])
    B = C
    cost, path = DTW(A, B, window = 4)
    print('Total Distance is ', cost)
    import matplotlib.pyplot as plt
    offset = 5
    plt.xlim([-1, max(len(A), len(B)) + 1])
    plt.plot(A)
    plt.plot(B + offset)
    for (x1, x2) in path:
        plt.plot([x1, x2], [A[x1], B[x2] + offset])
    plt.show()

if __name__ == '__main__':
    main()