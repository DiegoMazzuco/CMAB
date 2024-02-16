import numpy as np

from .MAB import MAB
from .Rewards import BernoulliFeature


class LinUCB(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, user_amount)
        self.d = d

        self.A = np.dstack([np.identity(d)] * user_amount) * lamb
        self.b = np.dstack([np.zeros([d, 1])] * user_amount)
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        A_inv = np.linalg.inv(self.A[:, :, user_id])
        theta = np.dot(A_inv, self.b[:, :, user_id])
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        self.A[:, :, user_id] += np.dot(x, x.T)
        self.b[:, :, user_id] += reward * x
