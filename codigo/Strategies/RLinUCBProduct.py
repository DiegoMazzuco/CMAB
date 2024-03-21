import numpy as np

from .MAB import MAB
from .Rewards import Reward


class RLinUCBProduct(MAB):
    def __init__(self, k: int, iters: int, reward_class: Reward, d: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, k)
        self.d = d
        random_start = 0.01

        self.xAinvx = np.zeros(k)
        self.A_inv = np.dstack([np.identity(d)] * k) / lamb
        self.theta = np.random.rand(d, 1, k) * random_start
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i)
        A_inv = self.A_inv[:, :, i]
        theta = self.theta[:, :, i]
        self.xAinvx[i] = x.T.dot(A_inv).dot(x)[0][0]
        aux = 0
        # Si A aumenta por ende inversa disminuye puede suceder que cuando A tiende a 0
        # la aproximacion de 0 negativo en vez de 0
        if self.xAinvx[i] > 0:
            aux = self.alpha * np.sqrt(self.xAinvx[i])
        return np.dot(theta.T, x) + aux

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i)
        A_inv = self.A_inv[:, :, i]
        theta = self.theta[:, :, i]

        k_n = A_inv.dot(x) / (1 + self.xAinvx[i])
        e_n = reward - x.T.dot(theta)[0, 0]

        self.A_inv[:, :, i] = A_inv - k_n.dot(x.T).dot(A_inv)
        self.theta[:, :, i] = theta + k_n * e_n
