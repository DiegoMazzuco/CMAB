import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.colors as colors


class Agglomerative():

    def __init__(self, ks, user_amount: int):
        self.x = None
        self.labels = np.zeros((user_amount, len(ks)))
        self.ks = ks

    def fit(self, x):
        link = linkage(pdist(x), method='complete')
        for i in range(len(self.ks)):
            self.labels[:, i] = fcluster(link, self.ks[i], criterion='maxclust')

        self.x = x

        return self.labels

    def get_labels(self):
        return self.labels

    def graph(self, labels=None):
        if labels is None:
            labels = self.labels

        for i in range(self.ks[0]):
            c = list(colors.cnames.values())[i + 120]
            filtered_label = self.x[labels == i + 1]
            plt.scatter(filtered_label[:, 0], filtered_label[:, 1], color=c)

        plt.show()


# Tests
if __name__ == '__main__':
    seed = random.randint(0, 1000)
    np.random.seed(seed)
    print('seed: ' + str(seed))
    data = np.random.rand(12, 2)
    data[0:4] = data[0:4] - [1, 1]
    data[4:8] = data[4:8] + [0.5, -0.5]

    clusters = Agglomerative([2, 3, 4], len(data)).fit(data)

    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 0])
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 1])
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 2])
    plt.show()
