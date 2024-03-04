import numpy as np

from .MAB import MAB
from .Rewards import BernoulliFeature


class RLinUCB(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.xAx = np.zeros(user_amount)
        self.A_inv = np.dstack([np.identity(d)] * user_amount) / lamb
        self.b = np.dstack([np.zeros([d, 1])] * user_amount)
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        A_inv = self.A_inv[:, :, user_id]
        theta = np.dot(A_inv, self.b[:, :, user_id])
        self.xAx[user_id] = x.T.dot(A_inv).dot(x)
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(self.xAx[user_id])

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        A_inv = self.A_inv[:, :, user_id]
        self.A_inv[:, :, user_id] = A_inv - A_inv.dot(x.dot(x.T)).dot(A_inv)/(1 + self.xAx[user_id])
        self.b[:, :, user_id] += reward * x
