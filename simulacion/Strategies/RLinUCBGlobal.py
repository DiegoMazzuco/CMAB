import numpy as np

from .MAB import MAB
from .Rewards import BernoulliFeature


class RLinUCBGlobal(MAB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, user_amount)

        self.d = d

        #(cI)^-1 = I/c
        self.xAinvx = 0
        self.A_inv = np.identity(d) / lamb
        self.theta = np.zeros([d, 1])
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))
        self.xAinvx = x.T.dot(self.A_inv).dot(x)
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(self.xAinvx)

        return p[0]

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i).reshape((self.d, 1))

        k_n = self.A_inv.dot(x)/( 1 + self.xAinvx)
        e_n = reward - x.T.dot(self.theta)[0, 0]

        self.A_inv = self.A_inv - k_n.dot(x.T).dot(self.A_inv)
        self.theta = self.theta + k_n * e_n
