import numpy as np
from scipy.spatial.distance import cdist
import random
np.random.seed(11)


class MyKmeans():
    def __init__(self, num_clusters):
        # randomly pick k rows of X ans initial centroids
        self.num_clusters = num_clusters

    def init_centroids(self, X):
        dimension_var = np.var(X, axis=0)
        cvmax_id = np.argmax(dimension_var)
        X_sort = X[X[:, cvmax_id].argsort()]
        centroid = np.zeros((self.num_clusters, X.shape[1]))
        num_subsets = X.shape[0] // self.num_clusters
        for i in range(self.num_clusters):
            start = i*num_subsets
            end = start + num_subsets
            if i == self.num_clusters - 1:
                end = X_sort.shape[0]
            centroid[i, :] = np.mean(X_sort[start:end], axis=0)
        self.centroids_list = [centroid]
        print(centroid)

    def assign_labels(self, X):
        # calculate pairwise distances btw data and centroids
        distances = cdist(X, self.centroids_list[-1])
        # return index of the closest centroid
        return np.argmin(distances, axis = 1)

    def update_centroids(self, X, labels):
        centroids = np.zeros((self.num_clusters, X.shape[1]))
        for cluster in range(self.num_clusters):
            # collect all points assigned to the k-th cluster
            X_cluster = X[labels == cluster, :]
            # take average
            centroids[cluster,:] = np.mean(X_cluster, axis = 0)
        return centroids

    def is_converged(self, new_centroids):
        # return True if two sets of centroids as the same
        if (new_centroids == []):
            return False
        return (set([tuple(a) for a in self.centroids_list[-1]]) == set([tuple(a) for a in new_centroids]))

    def fit(self, X):
        self.init_centroids(X)
        labels = []
        new_centroids = []

        while self.is_converged(new_centroids) == False:
            labels.append(self.assign_labels(X))
            new_centroids = self.update_centroids(X, labels[-1])
            self.centroids_list.append(new_centroids)

        return (self.centroids_list[-1], labels)

    def predict(self, X):
        distances = cdist(X, self.centroids_list[-1])
        # return index of the closest centroid
        return np.argmin(distances, axis = 1)

