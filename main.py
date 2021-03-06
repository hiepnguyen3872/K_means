from k_mean import MyKmeans
import matplotlib.pyplot as plt
import numpy as np
import time

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def kmeans_display(X, label, title = 'title'):
        K = np.amax(label) + 1
        X0 = X[label == 0, :]
        X1 = X[label == 1, :]
        X2 = X[label == 2, :]
        
        # print ("type: " + str(type(X0)))

        plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
        plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
        plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

        plt.axis('equal')
        plt.plot()
        
        plt.title(title)

        plt.show()


my_k_mean = MyKmeans(K)
(centroids, labels) = my_k_mean.fit(X)
print('Centers found by our algorithm:')
print(centroids)
kmeans_display(X, labels[-1])
print(my_k_mean.predict(X[:10]))



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
print(kmeans.predict(X[:10]))


#===========================================================    Luan test
print (labels)
kmeans_display(X, labels, 'our kmeans')
