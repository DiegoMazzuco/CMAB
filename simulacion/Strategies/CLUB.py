import math

import numpy as np

from .MAB import MAB
from .Rewards import BernoulliFeature
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


class CLUB(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 cluster_initial_start=0, cluster_iteration_ex=1000, cluster_mix_rew_start=-1, lamb=1,
                 alpha_2=1):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.A = np.dstack([np.identity(d)] * user_amount) * lamb
        self.b = np.dstack([np.zeros([d, 1])] * user_amount)
        self.thetas = np.zeros([d, user_amount])
        self.alpha = alpha
        self.cluster = np.zeros(user_amount)
        self.cluster_initial_start = cluster_initial_start
        self.cluster_iteration_ex = cluster_iteration_ex
        self.cluster_mix_rew_start = cluster_mix_rew_start
        self.iteration = 0

        self.Graph = np.ones([user_amount, user_amount])
        self.clusters = []
        g = csr_matrix(self.Graph)
        n_components, component_list = connected_components(g)
        self.clusters = component_list
        self.CBPrime = np.ones(user_amount)
        self.alpha_2 = alpha_2
        self.counter = np.zeros(user_amount)
        self.lamb = lamb

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        cluster_index = [i for i, elem in enumerate(self.clusters) if elem == self.clusters[user_id]]

        A = np.identity(self.d) * self.lamb
        b = np.zeros([self.d, 1])
        for cl_id in cluster_index:
            A += self.A[:, :, cl_id] - np.identity(self.d) * self.lamb
            b += self.b[:, :, cl_id]
        A_inv = np.linalg.inv(A)
        theta = np.dot(A_inv, b)[:, 0]
        # en club se agrega un logaritmo al final con la cantidad iteraciones
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x))) * np.sqrt(math.log10(self.iteration + 1))

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        self.__reward_update_one_theta(reward, x, user_id, 1)

        # Se actualiza el cluster
        self.updateGraphClusters(user_id)

        self.iteration += 1

    def __reward_update_one_theta(self, reward, x, user_id, factor_crecimiento):
        # Check if this useful
        self.A[:, :, user_id] += np.dot(x, x.T) * factor_crecimiento
        self.b[:, :, user_id] += reward * x * factor_crecimiento
        A_inv = np.linalg.inv(self.A[:, :, user_id])
        self.thetas[:, user_id] = np.dot(A_inv, self.b[:, :, user_id])[:, 0]

        self.counter[user_id] += 1
        counter = self.counter[user_id]
        self.CBPrime[user_id] = self.alpha_2 * np.sqrt(float(1 + math.log10(1 + counter))
                                                  / float(1 + counter))


    def updateGraphClusters(self, user_id):
        for j in range(self.user_amount):
            ratio = float(np.linalg.norm(self.thetas[:, user_id] - self.thetas[:, j], 2)) / float(
                self.CBPrime[user_id] + self.CBPrime[j])
            if ratio > 1:
                ratio = 0
            else:
                ratio = 1
            # print 'ratio',ratio
            self.Graph[user_id][j] = ratio
            self.Graph[j][user_id] = self.Graph[user_id][j]
        n_components, component_list = connected_components(csr_matrix(self.Graph))
        self.clusters = component_list
