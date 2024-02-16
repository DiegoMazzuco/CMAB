import numpy as np

from .MAB_OFF import MAB_OFF
from .Rewards import Reward


class EpsilonGreedy_OFF(MAB_OFF):
    def __init__(self, k: int, iters: int, reward_class: Reward, eps: float):
        super().__init__(k, iters, reward_class)
        self.eps = eps

    def calc_value(self, id):
        i = self.get_article_index(id)
        return self.k_reward[i]

    def select_action(self):
        p = np.random.rand()
        available_products = self.reward_class.get_available_products(self.eventIter)
        available_products = np.intersect1d(self.articles, available_products)
        if self.eps == 0 and self.n == 0:
            return np.random.choice(available_products)
        elif p < self.eps:
            # Randomly select an action
            return np.random.choice(available_products)
        else:
            # Take greedy action
            return super().select_action()
