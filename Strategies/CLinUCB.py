import numpy as np

from .Clusters import Kmeans
from .MAB import MAB
from .Rewards import BernoulliFeature


class CLinUCB(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 cluster_amount=2, cluster_it=50, cluster_initial_start = 1000, cluster_iteration_ex=1000,
                 cluster_mix_rew_start=-1,
                 lamb=1):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.A = np.dstack([np.identity(d)] * user_amount) * lamb
        self.b = np.dstack([np.zeros([d, 1])] * user_amount)
        self.thetas = np.zeros([d, user_amount])
        self.alpha = alpha
        self.kmeans = Kmeans(cluster_amount, cluster_it)
        self.cluster = np.zeros(k)
        self.cluster_initial_start = cluster_initial_start
        self.cluster_iteration_ex = cluster_iteration_ex
        self.cluster_mix_rew_start = cluster_mix_rew_start
        self.iteration = 0

    def get_kmeans(self):
        return self.kmeans

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        cluster_index = [i for i, elem in enumerate(self.cluster) if elem == self.cluster[user_id]]

        A = np.zeros((self.d, self.d))
        theta = np.zeros(self.d)
        for cl_id in cluster_index:
            A += self.A[:, :, cl_id]
            theta += self.thetas[:, cl_id]
        A = A / len(cluster_index)
        theta = theta / len(cluster_index)
        A_inv = np.linalg.inv(A)
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        self.__reward_update_one_theta(reward, x, user_id, 1)
        cluster_index = [i for i, elem in enumerate(self.cluster) if elem == self.cluster[user_id]]
        cluster_index.remove(user_id)
        #Aprendizaje entre todos
        if self.cluster_mix_rew_start != -1 and self.iteration > self.cluster_mix_rew_start:
            for cl_id in cluster_index:
                self.__reward_update_one_theta(reward, x, cl_id, 0.1)

        #Se actualiza el cluster
        if self.cluster_initial_start != -1 and self.iteration > self.cluster_initial_start and\
                self.iteration % self.cluster_iteration_ex == 0:
            self.cluster = self.kmeans.fit(self.thetas.T)
        self.iteration += 1

    def __reward_update_one_theta(self, reward, x, user_id, factor_crecimiento):
        self.A[:, :, user_id] += np.dot(x, x.T) * factor_crecimiento
        self.b[:, :, user_id] += reward * x * factor_crecimiento
        A_inv = np.linalg.inv(self.A[:, :, user_id])
        self.thetas[:, user_id] = np.dot(A_inv, self.b[:, :, user_id])[:, 0]
