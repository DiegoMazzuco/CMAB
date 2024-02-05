import numpy as np

from .MAB import MAB
from .Rewards import BernoulliFeature


class LinUCBGlobal(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.A = np.identity(d)
        self.b = np.zeros([d, 1])
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        A_inv = np.linalg.inv(self.A)
        theta = np.dot(A_inv, self.b)
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        self.A += np.dot(x, x.T)
        self.b += reward * x
