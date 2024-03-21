import numpy as np

from .MAB import MAB
from .Rewards import Reward
import math

class RLinUCBGlobal(MAB):
    def __init__(self, k: int, iters: int, reward_class: Reward, d: int, user_amount: int, alpha: float,
                 lamb: float):
        super().__init__(k, iters, reward_class, user_amount)

        self.d = d
        random_start = 0.01

        self.xAinvx = 0
        self.A_inv = np.identity(d) / lamb
        self.theta = np.random.rand(d, 1) * random_start
        self.alpha = alpha

    def calc_ucb(self, i, user_id):
        x = self.reward_class.get_feature(i)
        self.xAinvx = x.T.dot(self.A_inv).dot(x)[0][0]
        aux = 0
        # Si A aumenta por ende inversa disminuye puede suceder que cuando A tiende a 0
        # la aproximacion de 0 negativo en vez de 0
        if self.xAinvx > 0:
            aux = self.alpha * math.sqrt(self.xAinvx)
        return np.dot(self.theta.T, x)[0] + aux

    def reward_update(self, reward, i, user_id):
        x = self.reward_class.get_feature(i)

        k_n = self.A_inv.dot(x)/( 1 + self.xAinvx)
        e_n = reward - x.T.dot(self.theta)[0, 0]

        self.A_inv = self.A_inv - k_n.dot(x.T).dot(self.A_inv)
        self.theta = self.theta + k_n * e_n
