import numpy as np
import math

from matplotlib import pyplot as plt


class BernoulliFeature:

    def __init__(self, k: int, d: int, user_amount: int, cluster_amount,
                 max_prob = 0.5, epsilon=0.01, cluster_thetas = None,
                 epsilon_cluster=0.01):
        self.contextos = np.zeros((k, d))
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
        if cluster_thetas is not None and len(cluster_thetas) != cluster_amount:
            raise Exception("La cantidad de thetas de cluster debe ser la misma que la cantidad clusters")
        self.cluster_thetas = cluster_thetas
        self.init()

    def reset(self):
        self.init()

    def get_cluster_thetas(self):
        if self.cluster_thetas is None:
            cluster_thetas = np.zeros((self.cluster_amount , self.d))
            for i in range(self.cluster_amount):
                cluster_theta = np.random.randn(self.d)
                cluster_thetas[i] = cluster_theta/np.linalg.norm(cluster_theta)
            self.cluster_thetas = cluster_thetas
        return self.cluster_thetas

    def init_user(self):
        cluster_size = math.ceil(self.userAmount/self.cluster_amount)
        cluster_thetas = self.get_cluster_thetas()
        for i in range(self.cluster_amount):
            cluster_theta = cluster_thetas[i]
            for k in range(cluster_size):
                count = i*cluster_size + k
                if count >= self.userAmount:
                    continue
                theta = cluster_theta + self.epsilon_cluster * np.random.randn(self.d)
                theta /= np.linalg.norm(theta)
                self.best_thetas[count, :] = theta
                theta += self.epsilon * np.random.randn(self.d)
                theta /= np.linalg.norm(theta)
                self.thetas[count, :] = theta

    def init_contexto(self):
        for i in range(self.k):
            contexto = np.random.randn(self.d)
            contexto /= np.linalg.norm(contexto)
            self.contextos[i, :] = contexto

    def init_probabilities(self):
        for u in range(self.userAmount):
            for i in range(self.k):
                self.probablities[u, i] = np.dot(self.best_thetas[u], self.contextos[i])

    def init(self):
        self.init_user()
        self.init_contexto()
        self.init_probabilities()

    def get_contextos(self):
        return self.contextos

    def get_feature(self, a):
        return self.contextos[a]

    def get_theta(self):
        return self.thetas

    def get_best_thetas(self):
        return self.best_thetas

    def graph_best(self):
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.scatter(self.best_thetas[:, 0], self.best_thetas[:, 1])
        plt.show()

    def graph(self):
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.scatter(self.thetas[:, 0], self.thetas[:, 1])
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
        probability = np.dot(self.best_thetas[user_id], self.contextos[a]) * self.max_prob
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
