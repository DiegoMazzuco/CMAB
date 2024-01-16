import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.colors as colors


class Agglomerative():

    def __init__(self, ks, user_amount=10):
        self.x = None
        self.labels = np.zeros((user_amount, len(ks)))
        self.label = np.zeros(user_amount)
        self.ks = ks

        # greedy
        self.eps = 0.1
        # Number of arms
        self.k = len(ks)
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(self.k)
        self.k_reward = np.zeros(self.k)
        self.a = 0

    def fit(self, x):
        link = linkage(pdist(x), method='complete')
        for i in range(len(self.ks)):
            self.labels[:, i] = fcluster(link, self.ks[i], criterion='maxclust')

        self.x = x

    def get_new_label(self):
        r = np.random.rand()
        k = len(self.ks)
        if self.eps == 0 and self.n == 0:
            self.a = np.random.choice(k)
        elif r < self.eps:
            # Randomly select an action
            self.a = np.random.choice(k)
        else:
            # Take greedy action
            self.a = np.argmax(self.k_reward)

        self.label = self.labels[:, self.a]

    def update(self, reward):

        # Update counts
        self.n += 1
        self.k_n[self.a] += 1


        # Update results for a_k
        self.k_reward[self.a] = self.k_reward[self.a] + (
                reward - self.k_reward[self.a]) / self.k_n[self.a]

        self.get_new_label()

    def get_labels(self):
        return self.label

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

    clusters = Agglomerative([2, 3, 4]).fit(data)

    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 0])
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 1])
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=clusters[:, 2])
    plt.show()
