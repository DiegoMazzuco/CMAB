import numpy as np

from .MAB import MAB
from .Rewards import BernoulliFeature


class WoodburyLinUCBGlobal(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, user_amount)

        self.d = d

        # (cI)^-1 = I/c
        self.xAx = 0
        self.A_inv = np.identity(d) / lamb
        self.b = np.zeros([d, 1])
        self.theta = np.zeros([d, 1])
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i)
        self.theta = np.dot(self.A_inv, self.b)
        self.xAx = x.T.dot(self.A_inv).dot(x)
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(self.xAx)

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i)
        self.A_inv = self.A_inv - self.A_inv.dot(x.dot(x.T)).dot(self.A_inv) / (1 + self.xAx)
        self.b += reward * x
