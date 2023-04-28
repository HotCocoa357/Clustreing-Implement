# 自定义的KMeans聚类对象

import numpy as np


class myKMeans:
    def __init__(self, k=3, maxIter=20, distanceType = 'euclidean'):
        self.k = k
        self.maxIter = maxIter
        self.distanceType = distanceType

    def fit(self, X):
        # Initialize center randomly
        self.centers = X[np.random.choice(len(X), size=self.k, replace=False)]
        for i in range(self.maxIter):
            labels = self.predict(X)
            # Renew center with mean value of clusters
            newCenters = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])
            # Break when too close to Renew
            if np.allclose(self.centers, newCenters):
                break
            self.centers = newCenters

    # Assign point to the nearest center
    def predict(self, X):
        if self.distanceType == 'euclidean':
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers, axis=2)
        elif self.distanceType == 'manhattan':
            distances = np.sum(np.abs(X[:, np.newaxis, :] - self.centers), axis=2)
        else:
            raise ValueError('Invalid distance type')
        return np.argmin(distances, axis=1)
