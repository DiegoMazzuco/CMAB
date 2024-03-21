import numpy as np

from .MAB import MAB
from .Rewards import Reward


class RLinUCB(MAB):
    def __init__(self, k: int, iters: int, reward_class: Reward, d: int, user_amount: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d
        random_start = 0.01

        self.xAinvx = np.zeros(user_amount)
        self.A_inv = np.dstack([np.identity(d)] * user_amount) / lamb
        self.theta = np.random.rand(d, 1, user_amount) * random_start
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i)
        A_inv = self.A_inv[:, :, user_id]
        theta = self.theta[:, :, user_id]
        self.xAinvx[user_id] = x.T.dot(A_inv).dot(x)[0][0]
        aux = 0
        # Si A aumenta por ende inversa disminuye puede suceder que cuando A tiende a 0
        # la aproximacion de 0 negativo en vez de 0
        if self.xAinvx[user_id] > 0:
            aux = self.alpha * np.sqrt(self.xAinvx[user_id])
        else:
            self.xAinvx[user_id] = 0
        return np.dot(theta.T, x) + aux

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i)
        A_inv = self.A_inv[:, :, user_id]
        theta = self.theta[:, :, user_id]

        k_n = A_inv.dot(x) / (1 + self.xAinvx[user_id])
        e_n = reward - x.T.dot(theta)[0, 0]

        self.A_inv[:, :, user_id] = A_inv - k_n.dot(x.T).dot(A_inv)
        self.theta[:, :, user_id] = theta + k_n * e_n
