import matplotlib.pyplot as plt
import numpy as np
import math

from .Clusters import Agglomerative
from .MAB import MAB
from .Rewards import BernoulliFeature
import matplotlib.colors as colors


class RCLinUCB(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 clusters_amounts, lamb=1,
                 cluster_initial_start=1000, cluster_iteration_ex=1000, best_option_iteration=100,
                 cluster_mix_rew_start=-1):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.lamb = lamb
        self.A = np.dstack([np.identity(d)] * user_amount) * lamb
        self.A_inv = np.dstack([np.identity(d)] * user_amount) / lamb
        self.b = np.dstack([np.zeros([d, 1])] * user_amount)
        self.thetas = np.zeros([d, user_amount])

        # clusters
        self.models_amount = len(clusters_amounts)
        max_clust_amount = max(clusters_amounts)
        self.clusters_amounts = clusters_amounts
        self.theta_clust = np.zeros([d, self.models_amount, max_clust_amount])
        A_inv_clust = np.dstack([np.identity(d)] * self.models_amount * max_clust_amount) / lamb
        self.A_inv_clust = A_inv_clust.reshape((d,d,self.models_amount,max_clust_amount))

        self.alpha = alpha
        self.cluster = np.zeros(user_amount)
        self.cluster_initial_start = cluster_initial_start
        self.cluster_iteration_ex = cluster_iteration_ex
        self.cluster_mix_rew_start = cluster_mix_rew_start
        self.iteration = 0

        self.model = Agglomerative(clusters_amounts, user_amount)
        self.RC = Reward_Por_Cluster(clusters_amounts, iters, d)
        self.best_option = 0
        self.best_option_iteration = best_option_iteration

    def get_model(self):
        return self.model

    def get_rc(self):
        return self.RC

    def calc_ucb(self, i, user_id):
        best_option = self.RC.best_option() if self.iteration % self.best_option_iteration == 0 else self.best_option
        return self.calc_ucb_cluster(i, user_id, best_option)

    def calc_ucb_cluster(self, i, user_id, model_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        cluster_id = self.model.get_labels()[user_id, model_id]

        A_inv = self.A_inv_clust[:, :, model_id, cluster_id]
        theta = self.theta_clust[:, model_id, cluster_id]
        xAinvx = x.T.dot(A_inv).dot(x)[0][0]
        return np.dot(theta.T, x)[0] + self.alpha * math.sqrt(xAinvx)


    def calc_probabilities(self, i, user_id, model_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        cluster_id = self.model.get_labels()[user_id, model_id]

        p = np.dot(self.theta_clust[:, model_id, cluster_id].T, x)

        return p[0]

    def reward_update(self, reward, i, user_id):

        probs = np.zeros(self.models_amount)
        for j in range(self.models_amount):
            probs[j] = self.calc_probabilities(i, user_id, j)

        self.RC.update(reward, probs)

        self.__reward_update_one_theta(reward, user_id, i)

        # Se actualiza el cluster
        if self.cluster_initial_start != -1 and self.iteration > self.cluster_initial_start and \
                self.iteration % self.cluster_iteration_ex == 0:
            self.model.fit(self.thetas.T)
            for model_id in range(self.models_amount):
                for cluster_id in range(self.clusters_amounts[model_id]):
                    self.__recalculate_clusters(cluster_id, model_id)
        else:
            for model_id in range(self.models_amount):
                self.__reward_update_cluster_theta(reward, user_id, i, model_id)
        self.iteration += 1

    def __recalculate_clusters(self, cluster_id, model_id):
        labels = self.model.get_labels()[:, model_id]
        cluster_index = [i for i, elem in enumerate(labels) if elem == cluster_id]

        A = np.identity(self.d) * self.lamb
        b = np.zeros([self.d, 1])
        for cl_id in cluster_index:
            A += self.A[:, :, cl_id] - np.identity(self.d) * self.lamb
            b += self.b[:, :, cl_id]
        self.A_inv_clust[:, :, model_id, cluster_id] = np.linalg.inv(A)
        self.theta_clust[:, model_id, cluster_id] = np.dot(self.A_inv_clust[:, :, model_id, cluster_id], b)[:, 0]

    def __reward_update_one_theta(self, reward, user_id, i):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        self.A[:, :, user_id] += np.dot(x, x.T)
        self.b[:, :, user_id] += reward * x

        A_inv = self.A_inv[:, :, user_id]
        theta = self.thetas[:, user_id]
        xAinvx = x.T.dot(A_inv).dot(x)

        k_n = A_inv.dot(x) / (1 + xAinvx)
        e_n = reward - x.T.dot(theta)[0]  # it is matrix 1x1

        self.A_inv[:, :, user_id] = A_inv - k_n.dot(x.T).dot(A_inv)
        self.thetas[:, user_id] = theta + k_n.reshape(self.d) * e_n

    def __reward_update_cluster_theta(self, reward, user_id, i, model_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        cluster_id = self.model.get_labels()[user_id, model_id]

        A_inv = self.A_inv_clust[:, :, model_id, cluster_id]
        theta = self.theta_clust[:, model_id, cluster_id]
        xAinvx = x.T.dot(A_inv).dot(x)

        k_n = A_inv.dot(x) / (1 + xAinvx)
        e_n = reward - x.T.dot(theta)[0]

        self.A_inv_clust[:, :, model_id, cluster_id] = A_inv - k_n.dot(x.T).dot(A_inv)
        self.theta_clust[:, model_id, cluster_id] = theta + k_n.reshape(self.d) * e_n


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
            reward = reward / np.max(probs)
        for i in range(self.cluster_amount):
            # Actualizar resultado por cada cluster
            self.k_reward[i] = self.k_reward[i] + (reward * probs[i] - self.k_reward[i]) / self.n
            self.k_rewards[i, self.n - 1] = self.k_reward[i]
        self.best_options[self.n - 1] = self.clusters_amounts[self.best_option()]
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
        plt.ylim(0, max(self.clusters_amounts) + 1)

        plt.show()
