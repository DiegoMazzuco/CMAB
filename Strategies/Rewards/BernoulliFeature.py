import numpy as np
import math

from matplotlib import pyplot as plt


class BernoulliFeature:

    def __init__(self, k: int, d: int, user_amount: int, cluster_amount, max_prob = 0.5, epsilon=0.01,
                 epsilon_cluster=0.01):
        self.products = np.zeros((k, d))
        self.probablities = np.zeros((user_amount, k))
        self.best_thetas = np.zeros((user_amount, d))
        self.thetas = np.zeros((user_amount, d))
        self.d = d
        self.k = k
        self.userAmount = user_amount
        self.cluster_amount = cluster_amount
        self.epsilon = epsilon
        self.epsilon_cluster = epsilon_cluster
        self.max_prob = max_prob
        self.init()

    def reset(self):
        self.init()

    def init_user(self):
        cluster_size = math.ceil(self.userAmount/self.cluster_amount)
        for i in range(self.cluster_amount):
            cluster_thetas = np.random.randn(self.d)
            cluster_thetas /= np.linalg.norm(cluster_thetas)
            for k in range(cluster_size):
                count = i*cluster_size + k
                if count >= self.userAmount:
                    continue
                theta = cluster_thetas + self.epsilon_cluster * np.random.randn(self.d)
                theta /= np.linalg.norm(theta)
                self.best_thetas[count, :] = theta
                self.thetas[count, :] = theta + self.epsilon * np.random.randn(self.d)

    def init_articles(self):
        for i in range(self.k):
            product = np.random.randn(self.d)
            product /= np.linalg.norm(product)
            self.products[i, :] = product

    def init_probabilities(self):
        for u in range(self.userAmount):
            for i in range(self.k):
                self.probablities[u, i] = np.dot(self.best_thetas[u], self.products[i])

    def init(self):
        self.init_user()
        self.init_articles()
        self.init_probabilities()

    def get_products(self):
        return self.products

    def get_feature(self, a):
        return self.products[a]

    def get_theta(self):
        return self.thetas

    def get_best_thetas(self):
        return self.best_thetas

    def graph(self):

        plt.scatter(self.best_thetas[:, 0], self.best_thetas[:, 1])
        plt.show()

    def get_probabilities(self, user_id: int):
        return self.probablities[user_id, :] * self.max_prob

    def get_probability(self, user_id: int, a: int):
        return self.probablities[user_id, a] * self.max_prob

    def get_regret(self, user_id: int, reward: int):
        max_reward = max(self.get_probabilities(user_id)) * self.max_prob
        return max_reward - reward

    def get_reward(self, user_id: int, a: int):
        p = np.random.rand()
        #TODO: esta limitado entre 0 y 1 pero puede ser negativo
        probability = np.dot(self.thetas[user_id], self.products[a]) * self.max_prob
        if p < probability:
            return 1
        else:
            return 0


# Tests
if __name__ == '__main__':
    k_test = 10
    d_test = 4
    u_a = 100
    b = BernoulliFeature(k_test, d_test, u_a)
    for i_test in range(k_test):
        b.get_reward(i_test, i_test)
