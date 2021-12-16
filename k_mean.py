import numpy as np
from scipy.spatial.distance import cdist
import random
np.random.seed(11)


class MyKmeans():
    def __init__(self, num_clusters, stop_condition = 0.0001):
        # Init number of clusters
        self.num_clusters = num_clusters

        self.stop_condition = stop_condition
        self.last_centroids = []
        self.last_labels = []

    def init_centroids(self, X):
        # calculate the variance of data in each dimension
        dimension_var = np.var(X, axis=0)

        # get the dimension with maximum dimension
        cvmax_id = np.argmax(dimension_var)

        # Sort data in descending order
        X_sort = X[X[:, cvmax_id].argsort()]

        centroid = np.zeros((self.num_clusters, X.shape[1]))

        # calculate the number of observations per cluster
        num_observation = X.shape[0] // self.num_clusters

        #initialize centroid corresponds the median of observations
        for i in range(self.num_clusters):
            start = i*num_observation
            end = start + num_observation
            if i == self.num_clusters - 1:
                end = X_sort.shape[0]
            centroid[i, :] = np.mean(X_sort[start:end], axis=0)
        self.last_centroids = centroid
        print(centroid)


    def clustering(self, X, centroids):
        # calculate distance from data to centroids
        distances = cdist(X, centroids)

        # calculate labels assigned to data
        new_labels = np.argmin(distances, axis = 1)

        return new_labels


    def update_centroids(self, X, labels):
        centroids = np.zeros((self.num_clusters, X.shape[1]))

        for cluster in range(self.num_clusters):

            # get all data assigned to this cluster
            X_cluster = X[labels == cluster, :]

            # update centroids
            centroids[cluster,:] = np.mean(X_cluster, axis = 0)
        return centroids


    def is_converged(self, new_centroids, last_centroid):

        if (new_centroids == []):
            return False

        # return True if the difference between two successive centroids smaller than stop condition
        return np.linalg.norm(new_centroids - last_centroid) < self.stop_condition


    def fit(self, X):
        # Initialize centroids
        self.init_centroids(X)

        new_centroids = []

        while self.is_converged(new_centroids, self.last_centroids) == False:

            # update labels assign to data
            self.last_labelss = self.clustering(X, self.last_centroids)

            # calculate new centroids
            new_centroids = self.update_centroids(X, self.last_labelss)

            # add new centroids to centroids list
            self.last_centroids = new_centroids

        return (self.last_centroids, self.last_labelss)


    def predict(self, X):

        # calculate distance from data to centroids
        distances = cdist(X, self.last_centroids)

        # calculate labels assigned to data
        new_labels = np.argmin(distances, axis = 1)

        return new_labels

