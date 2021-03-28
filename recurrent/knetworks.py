import torch
import torch.nn

import math

from datetime import datetime, date, timedelta, time

from torch.utils.tensorboard import SummaryWriter

from lstm import LSTM

class knetworks:
    def __init__(self, k, data, vocabSize, device=torch.device("cpu")):
        super(knetworks, self).__init__()

        self.km = kmeans(k)
        self.k = k
        self.centroids = []

        self.device = device

        self.data = data

        self.D = []
        self.W = []

        self.vocabSize = vocabSize

        self.networks = []

        for _ in range(k):
            self.networks.append(LSTM(vocabSize, device=device))

    def sampleRandom(self, centroid):
        return np.random.choice(np.array(range(len(self.data))),p=self.W[centroid])

    def train(self, samples, epochs):
        for i in range(self.k):
            for s in range(samples):
                user = self.sampleRandom(i)
                print("[" + str(i) + "][" + str(s) + "->" + str(user) + "] Training..." , end="\r")
                self.networks[i].model.train()
                self.networks[i].train(self.data[user], epochs=epochs)
        print("\n")

    def calcMean(self, data):
        n = len(data)
        mean = np.empty((len(data[0])))
        for i in range(len(data[0])):
            mean[i] = np.sum(data[:n,i])
        return mean/n

    def fit(self, max_iters=1, optimize=False, verbose=False):
        means = []
        for user in self.data:
            means.append(self.calcMean(user))
        means = np.array(means)

        self.km.fit(means, max_iters=max_iters, optimize=optimize, verbose=verbose)
        self.k = self.km.k
        self.centroids = self.km.centroids

        self.D = self.km.calcDistances(self.centroids, means)

        self.W = np.minimum((1/(self.D+0.001)**2), np.full(self.D.shape, 50))

        self.W = np.array([self.W[i]/sum(self.W[i]) for i in range(self.k)])

    def save(self, filepath):
        # save the model state_dicts
        for i, net in enumerate(self.networks):
            torch.save(net.model.state_dict(), filepath + "/models/CN_" + str(i) + ".pth")

        # save the centroids array
        np.savetxt(filepath + '/centroids.csv', self.centroids, delimiter=',')
        # save the distances array
        np.savetxt(filepath + '/distances.csv', self.D, delimiter=',')
        # save the weights array
        np.savetxt(filepath + '/weights.csv', self.W, delimiter=',')


    def load(self, filepath):
        # load the model state_dicts
        for i,net in enumerate(self.networks):
            net.model.load_state_dict(torch.load(filepath + "/models/CN_" + str(i) + ".pth", map_location=self.device))

        # load the centroids array
        self.centroids = np.loadtxt(filepath + '/centroids.csv', delimiter=',')
        # load the distances array
        self.D = np.loadtxt(filepath + '/distances.csv', delimiter=',')
        # load the weights array
        self.W = np.loadtxt(filepath + '/weights.csv', delimiter=',')

        self.k = len(self.centroids)

    def predict(self, data, future=1):
        mean = self.calcMean(data)
        distances = self.km.calcDistances(self.centroids, mean)
        weights = np.minimum((1/distances**(distances*0.75)+0.0001), np.full(distances.shape, 50))
        weights = np.array([weights[i]/sum(weights) for i in range(self.k)])

        #print(distances, weights)

        prediction = []
        for i, net in enumerate(self.networks):
            net.model.eval()
            prediction.append(weights[i] * net.predict(data, future))

        return np.sum(np.array(prediction), axis=0)
