import math
import numpy as np

from .MAB_OFF import MAB_OFF
from .Rewards import Reward


class UCB_OFF(MAB_OFF):
    def __init__(self, k: int, iters: int, reward_class: Reward, alpha: float):
        self.alpha = alpha
        super().__init__(k, iters, reward_class)


    def calc_value(self, id):
        i = self.get_article_index(id)
        return self.k_reward[i] + self.alpha * math.sqrt(2 * math.log(self.n + 1) / (self.k_n[i] + 1))



