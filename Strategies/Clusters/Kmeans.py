import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class Kmeans:

    def __init_centroids(self, xs):
        centroids_idx = np.random.choice(range(len(xs)), 1, replace=False)

        for j in range(self.k-1):

            distance = np.zeros([j+1, len(xs)])
            for i in range(j+1):
                distance[i, :] = [np.linalg.norm(xs[centroids_idx[i]] - x) for x in xs]
            distance = np.min(distance, axis=0)
            new_centroid_id = np.argmax(distance)
            centroids_idx = np.append(centroids_idx, new_centroid_id)
        self.centroids = xs[centroids_idx]

    def __init__(self, k: int, max_iterations=500):
        self.xs = None
        self.labels = None
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, xs: np.array):
        self.__init_centroids(xs)

        j = 0

        old_labels = np.zeros(len(xs))
        labels = np.zeros(len(xs))
        while j < self.max_iterations:
            for i in range(len(xs)):
                distance = [np.linalg.norm(c - xs[i]) for c in self.centroids]
                labels[i] = np.argmin(distance)

            if np.array_equal(old_labels, labels):
                break
            old_labels = labels

            for i in range(self.k):
                self.centroids[i] = np.average(xs[np.where(labels == i)], axis=0)
            j += 1

        self.xs = xs
        self.labels = labels
        return labels

    def graph(self, labels=None):
        if labels is None:
            labels = self.labels

        for i in range(self.k):
            c = list(colors.cnames.values())[i+120]
            filtered_label = self.xs[labels == i]
            plt.scatter(filtered_label[:, 0], filtered_label[:, 1], color=c)
            plt.scatter(self.centroids[i, 0], self.centroids[i, 1], marker='x', color=c)

        plt.show()


# Tests
if __name__ == '__main__':
    seed = 681
    #seed = random.randint(0, 1000)
    np.random.seed(seed)
    print('seed: ' + str(seed))
    data = np.random.rand(12, 2)
    data[0:4] = data[0:4] - [1, 1]
    data[4:8] = data[4:8] + [2, -2]
    plt.scatter(data[:, 0], data[:,1])
    plt.show()
    kmeans = Kmeans(3)
    labels = kmeans.fit(data)

    kmeans.graph()
