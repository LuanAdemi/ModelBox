from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import random
import sys

import concurrent.futures

#   An implementation of the kmeans++ (https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) algorithm with
#   k optimization using the elbow method.


class kmeans:
    def __init__(self, k=2):
        super(kmeans, self).__init__()

        self.k = k

    def distance(self, p1, p2):
        return np.sum((p1 - p2)**2)

    def initCentroids(self, k, data):
        centroids = []
        centroids.append(data[np.random.randint(
            data.shape[0]), :])
        if self.verbose:
            print("Searching for best starting centroids...")
        for c_id in range(k-1):
            dist = []
            for point in data:
                d = sys.maxsize
                for j in range(len(centroids)):
                    temp_dist = self.distance(point, centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)

            dist = np.array(dist)
            next_centroid = data[np.argmax(dist), :]
            centroids.append(next_centroid)

            if self.verbose:
                print("[%d] Setting centroid to" % (c_id+2), centroids[c_id+1], end="\r")

            dist = []
        centroids = np.array(centroids)
        return centroids

    def calcDistances(self, centroids, data):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances

    def fit(self, data, max_iters=10, verbose=False, optimize=False):
        self.verbose = verbose
        if optimize == False:
            if self.verbose:
                print("Finding the best starting centroids...")
            centroids = self.initCentroids(self.k, data)

            for epoch in range(max_iters):
                if self.verbose:
                    print("[%d] KMean'in..." % epoch)
                distances = self.calcDistances(centroids, data)

                affiliations = np.argmin(distances, axis=0)

                centroids = np.array([data[affiliations==k].mean(axis=0) for k in range(centroids.shape[0])])

            self.centroids = centroids
            self.affiliations = affiliations

        else:
            def run(cK):
                centroids = self.initCentroids(cK, data)

                for epoch in range(max_iters):
                    distances = self.calcDistances(centroids, data)

                    affiliations = np.argmin(distances, axis=0)

                    centroids = np.array([data[affiliations==k].mean(axis=0) for k in range(centroids.shape[0])])

                silscore = silhouette_score(data, affiliations)
                if verbose:
                    print("[%d] Elbow'in... SilScore=%f" % (cK, silscore))

                return (silscore, cK, centroids, affiliations)

            maxSilscore = 0

            for cK in range(2, 20):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run, cK)
                    return_value = future.result()
                    silscore, K, centroids, aff = return_value

                    if silscore > maxSilscore:
                        self.centroids = centroids
                        self.k = K
                        self.affiliations = aff
