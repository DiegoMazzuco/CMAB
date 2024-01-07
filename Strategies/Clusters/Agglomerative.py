import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


class Agglomerative():

    def __init__(self, ks):
        self.ks = ks

    def fit(self, x):
        link = linkage(pdist(x), method='complete')
        labels = np.zeros((len(x), len(self.ks)))
        for i in range(len(self.ks)):
            labels[:, i] = fcluster(link, self.ks[i], criterion='maxclust')

        return labels[:, 0]


# Tests
if __name__ == '__main__':
    seed = random.randint(0, 1000)
    np.random.seed(seed)
    print('seed: ' + str(seed))
    data = np.random.rand(12, 2)
    data[0:4] = data[0:4] - [1, 1]
    data[4:8] = data[4:8] + [0.5, -0.5]

    clusters = Agglomerative([2, 3, 4]).fit(data)

    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 0])
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 1])
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 2])
    plt.show()
