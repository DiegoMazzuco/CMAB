import matplotlib.pyplot as plt
import numpy as np
import math

from .Clusters import Agglomerative
from .MAB import MAB
from .Rewards import BernoulliFeature
import matplotlib.colors as colors

class CLinUCB(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 clusters_amounts, lamb=1,
                 cluster_initial_start=1000, cluster_iteration_ex=1000, cluster_mix_rew_start=-1):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.lamb = lamb
        self.A = np.dstack([np.identity(d)] * user_amount) * lamb
        self.b = np.dstack([np.zeros([d, 1])] * user_amount)
        self.thetas = np.zeros([d, user_amount])
        self.alpha = alpha
        self.cluster = np.zeros(user_amount)
        self.cluster_initial_start = cluster_initial_start
        self.cluster_iteration_ex = cluster_iteration_ex
        self.cluster_mix_rew_start = cluster_mix_rew_start
        self.iteration = 0

        self.len_c = len(clusters_amounts)
        self.model = Agglomerative(clusters_amounts, user_amount)
        self.RC = Reward_Por_Cluster(clusters_amounts, iters, d)

    def get_model(self):
        return self.model

    def get_rc(self):
        return self.RC

    def calc_ucb(self, i, user_id):
        return self.calc_ucb_cluster(i, user_id, self.RC.best_option())
    def calc_ucb_cluster(self, i, user_id, cluster_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        labels = self.model.get_labels()[:, cluster_id]
        cluster_index = [i for i, elem in enumerate(labels) if elem == labels[user_id]]

        A = np.identity(self.d) * self.lamb
        b = np.zeros([self.d, 1])
        for cl_id in cluster_index:
            A += self.A[:, :, cl_id] - np.identity(self.d) * self.lamb
            b += self.b[:, :, cl_id]
        A_inv = np.linalg.inv(A)
        theta = np.dot(A_inv, b)[:, 0]
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p[0]

    def calc_probabilities(self, i, user_id, cluster_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        labels = self.model.get_labels()[:, cluster_id]
        cluster_index = [i for i, elem in enumerate(labels) if elem == labels[user_id]]

        A = np.identity(self.d) * self.lamb
        b = np.zeros([self.d, 1])
        for cl_id in cluster_index:
            A += self.A[:, :, cl_id] - np.identity(self.d) * self.lamb
            b += self.b[:, :, cl_id]
        A_inv = np.linalg.inv(A)
        theta = np.dot(A_inv, b)[:, 0]
        p = np.dot(theta.T, x)

        return p[0]

    def reward_update(self, reward, i, user_id):

        x = self.reward_class.get_feature(i).reshape((self.d, 1))

        # if True:
        #     labels = self.model.get_labels()[:, self.RC.best_option()]
        #     cluster_index = [i for i, elem in enumerate(labels) if elem == labels[user_id] and elem != user_id]
        #     for i in cluster_index:
        #         self.__reward_update_one_theta(reward, x, i, 0.1)

        self.__reward_update_one_theta(reward, x, user_id, 1)

        probs = np.zeros(self.len_c)
        for j in range(self.len_c):
            probs[j] = self.calc_probabilities(i, user_id, j)

        self.RC.update(reward, probs)

        # Se actualiza el cluster
        if self.cluster_initial_start != -1 and self.iteration > self.cluster_initial_start and \
                self.iteration % self.cluster_iteration_ex == 0:
            self.model.fit(self.thetas.T)
        self.iteration += 1

    def __reward_update_one_theta(self, reward, x, user_id, factor_crecimiento):
        #Check if this useful
        self.A[:, :, user_id] += np.dot(x, x.T) * factor_crecimiento
        self.b[:, :, user_id] += reward * x * factor_crecimiento
        A_inv = np.linalg.inv(self.A[:, :, user_id])
        theta = np.dot(A_inv, self.b[:, :, user_id])[:, 0]
        self.thetas[:, user_id] = theta


class Reward_Por_Cluster:

    def __init__(self, clusters_amounts, iters, d):
        # Step count
        self.n = 1
        # cluster amount
        self.clusters_amounts = clusters_amounts
        self.cluster_amount = len(clusters_amounts)
        # Step count for each arm
        self.k_reward = np.ones(len(clusters_amounts))
        self.k_rewards = np.ones((len(clusters_amounts), iters))
        self.best_options = np.zeros((iters))
        self.d = d

    def update(self, reward, probs):
        if np.max(probs) > 0:
            reward = reward/np.max(probs)
        for i in range(self.cluster_amount):
            # Actualizar resultado por cada cluster
            self.k_reward[i] = self.k_reward[i] + (reward * probs[i] - self.k_reward[i]) / self.n
            self.k_rewards[i, self.n-1] = self.k_reward[i]
        self.best_options[self.n-1] = self.clusters_amounts[self.best_option()]
        # Update counts
        self.n += 1

    def best_option(self):
        # Aplico AIC, para penalizar los algoritmos mas complejos con mas clusters
        options = np.zeros(self.cluster_amount)
        for i in range(self.cluster_amount):
            if self.k_reward[i] != 0:
                options[i] = -math.log(self.k_reward[i]) + self.clusters_amounts[i] * self.d * math.log(self.n) / self.n
            else:
                # En caso de ser 0 reward se pone como Nan para ser ignorado por el np.nanargmin
                options[i] = np.nan
        # Puede fallar si todos reward son 0 y por ende las options son NaN
        try:
            return np.nanargmin(options)
        except:
            # En caso que todos sean reward 0 se devuelve el que tenga menos cantidad cluster
            return np.argmin(self.clusters_amounts)

    def graph(self):
        plt.figure()
        for i in range(self.cluster_amount):
            c = list(colors.cnames.values())[i + 120]
            plt.plot(self.k_rewards[i, :], label="cluster" + str(self.clusters_amounts[i]), color=c)
        plt.legend(bbox_to_anchor=(1.3, 0.5))
        plt.show()
        plt.figure()
        plt.plot(self.best_options, label="best option")
        plt.ylim(0, max(self.clusters_amounts)+1)

        plt.show()
