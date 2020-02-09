def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import statsmodels
import sklearn
import sys
import timeit
from math import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.grid_search import *
from sklearn.learning_curve import *
from sklearn.model_selection import *
from sklearn.manifold import *
from sklearn.metrics import *
from scipy.cluster.hierarchy import *
from matplotlib.collections import *
from sklearn.decomposition import *
from sklearn.cross_decomposition import *


np.random.seed(42)
pd.set_option('display.max_columns', 51)
np.set_printoptions(precision=5, suppress=True)


#########################################################
def feature_selection_classification(x_train, y_train, columns):
    return


#########################################################
def feature_selection_regression(x_train, y_train, columns):

    ranks = pd.DataFrame(data={'Feature': columns})

    def rank_to_dict(metrics, order=1):
        minmax = MinMaxScaler()
        scaled_metrics = minmax.fit_transform(order*np.array(metrics).T)
        return scaled_metrics

    ridge = RidgeCV()
    ridge.fit(x_train, y_train)
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_))

    bridge = BayesianRidge()
    bridge.fit(x_train, y_train)
    ranks["Bayesian Ridge"] = rank_to_dict(np.abs(bridge.coef_))

    lasso = LassoCV(n_jobs=-1)
    lasso.fit(x_train, y_train)
    ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_.T))

    rlasso = RandomizedLasso(alpha='bic')
    rlasso.fit(x_train, y_train)
    ranks["Randomized Lasso"] = rank_to_dict(np.abs(rlasso.scores_))

    rfe = RFE(lr, n_features_to_select=2)
    rfe.fit(x_train, y_train)
    ranks["RFE"] = rank_to_dict(rfe.ranking_, order=-1)

    rfe = RFECV(lr, n_jobs=-1)
    rfe.fit(x_train, y_train)
    ranks["RFECV"] = rank_to_dict(rfe.ranking_, order=-1)

    rf = RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=-1)
    rf.fit(x_train, y_train)
    ranks["RF"] = rank_to_dict(rf.feature_importances_)

    f, pval = f_regression(x_train, y_train, center=True)
    ranks["Correlation"] = rank_to_dict(f)

    ranks["Mean"] = ranks.mean(axis=1)
    ranks["Median"] = ranks.median(axis=1)
    ranks["Std"] = ranks.std(axis=1)


#########################################################
class recursive_outlier_elimination():
    def __init__(self, model, quantile=.95, floor_train_size=.9):
        self.quantile = quantile
        self.floor_train_size = floor_train_size
        self.remaining_rows = []
        self.model = model
        self.best_model = []

    def fit(self, train, response_variable):
        best_score = sys.maxsize

        self.remaining_rows = train.index

        while True:
            x = train.ix[self.remaining_rows]
            y = response_variable.ix[self.remaining_rows]

            self.model.fit(x, y)
            score = metrics.mean_squared_error(self.model.predict(x), y)
            if score < best_score:
                best_score = score

                residuals = y - self.model.predict(x)
                self.remaining_rows = residuals[abs(residuals) <= abs(residuals).quantile(self.quantile)].index
                self.best_model = self.model

                if len(self.remaining_rows) < len(train) * self.floor_train_size:
                    break
            else:
                self.best_model = self.model
                break

    def predict(self, test):
        return self.best_model.predict(test)


#########################################################
def forward_selection(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0

    while remaining and current_score == best_new_score:

        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            score = statsmodels.formula.api.smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()

        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
    model = statsmodels.formula.api.smf.ols(formula, data).fit()

    return model


#########################################################
def find_closest_series(x, y, verbose=False):
    if len(x) > len(y):
        raise ValueError('x is longer than y')
    elif len(x) == len(y):
        return y, 0
    else:
        best_indices = []
        min_distance = sys.maxsize
        best_match = np.array([])
        best_prediction = sys.maxsize
        for i in np.arange(len(y) - len(x)):
            distance = DTWDistance(x, y[i:i + len(x)], 1)

            if verbose:
                print(str(i) + ', ' + str(distance))

            if distance < min_distance:
                min_distance = distance
                best_match = y[i:i + len(x)]
                best_indices = list(range(i, i + len(x)))
                best_prediction = y[i + len(x) + 1] - y[i + len(x)]

        return best_match, min_distance, best_indices, best_prediction


#########################################################
def DTWDistance(s1, s2, w):

    DTW = {}
    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


#########################################################
def DTW_distance_matrix(df, w):
    columns = list(df.columns)
    n = len(columns)

    distance = np.zeros([n, n])
    for i in np.arange(n):
        for j in np.arange(0, i):
            distance[i][j] = DTWDistance(df[columns[i]].values, df[columns[j]].values, w)

    return distance


#########################################################
def time_series_cv(x_train, y_train, number_folds=4, method='increasing'):

    k = int(np.floor(float(x_train.shape[0]) / number_folds))
    print('Size of each fold: ', k)

    accuracies = np.zeros(number_folds - 1)
    periods = []

    for i in range(2, number_folds + 1):

        if method == 'sliding':

            split = 0.5
            x = x_train[(k*(i-2)):(k*i)]
            y = y_train[(k*(i-2)):(k*i)]

            periods.append([(k*(i-2)), (k*i)])

            index = int(np.floor(x.shape[0] * split))

            x_train_fold = x[:index]
            y_train_fold = y[:index]

            x_test_fold = x[index:]
            y_test_fold = y[index:]

        elif method == 'increasing':

            split = float(i - 1) / i
            x = x_train[:(k*i)]
            y = y_train[:(k*i)]

            periods.append([0, k*i])

            index = int(np.floor(x.shape[0] * split))

            x_train_fold = x[:index]
            y_train_fold = y[:index]

            x_test_fold = x[index:]
            y_test_fold = y[index:]

        elif method == 'fixed':

            x_train_fold = x_train[(k*(i-2)):(k*(i-1))]
            y_train_fold = y_train[(k*(i-2)):(k*(i-1))]

            x_test_fold = x_train[-k-1:]
            y_test_fold = y_train[-k-1:]

            periods.append([(k*(i-2)), (k*(i-1))])

        else:
            raise ValueError('method unknown')

        print('Size of train + test: ', x.shape)
        print('Size of train: ', x_train_fold.shape)
        print('Size of test:  ', x_test_fold.shape)

        model = linear_model.LinearRegression(normalize=True)
        model.fit(x_train_fold, y_train_fold)
        y_predicted = model.predict(x_test_fold)

        accuracies[i - 2] = r_score(y_test_fold, y_predicted) * 100

    return accuracies, periods


#########################################################
def pca_get_components(df, min_explained_variance=0.95):
    n = df.shape[1]
    pca = sklearn.decomposition.PCA(n_components=n)
    pca.fit(df)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(explained_variance > min_explained_variance)[0][0]

    return n_components + 1


#########################################################
def adversarial_selection(x_train, x_test):
    y_train = np.zeros(x_train.shape[0])
    y_test = np.ones(x_test.shape[0])
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

#    classifier = RandomForestClassifier(n_estimators=2000, max_depth=3, n_jobs=-1)
    classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=100, weights='distance', n_jobs=-1)
    cv = sklearn.model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5678)
    y_predicted = sklearn.model_selection.cross_val_predict(classifier, x, y, cv=cv, n_jobs=-1, method='predict_proba')

    confusion_matrix(y, y_predicted.argmax(axis=1))
    auc = sklearn.metrics.roc_auc_score(y, y_predicted[:, 1])
    print("# AUC: {:.2%}".format(auc))
    logloss = sklearn.metrics.log_loss(y, y_predicted)
    print("# Logloss: {:.2%}".format(logloss))

    sorted_indices = y_predicted[:, 1].argsort()
    y_sorted = y[sorted_indices]
    predictions_sorted = y_predicted[sorted_indices]
    predictions_sorted_train = predictions_sorted[y_sorted == 0]
    predictions_sorted_test = predictions_sorted[y_sorted == 1]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(predictions_sorted_train)
    ax1.axhline(y=.5, linewidth=2, color='orange')
    ax1.set_xlabel('')
    ax1.set_title('Adversarial validation in train set (0)')
    ax2 = fig.add_subplot(212)
    ax2.plot(predictions_sorted_test)
    ax2.axhline(y=.5, linewidth=2, color='orange')
    ax2.set_title('Adversarial validation in test set (1)')
    plt.show()

    plt.figure(2)
    plt.plot(predictions_sorted_train[:, 1])
    plt.plot(predictions_sorted_test[:, 1])
    plt.axhline(y=.5, linewidth=2, color='orange')
    plt.show()

    return sorted_indices


#########################################################
def mean_decrease_accuracy(X, Y, model, metric=r2_score):

    scores = collections.defaultdict(list)

    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        model.fit(X_train, Y_train)
        accuracy = metric(Y_test, model.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuffled_accuracy = metric(Y_test, model.predict(X_t))
            scores[str(i)].append((accuracy-shuffled_accuracy)/accuracy)

    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
