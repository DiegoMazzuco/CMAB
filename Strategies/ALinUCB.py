import numpy as np

from .CLinUCB import CLinUCB
from .Clusters import Agglomerative
from .Rewards import BernoulliFeature


class ALinUCB(CLinUCB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                 cluster_initial_start=2000, cluster_iteration_ex=1000,
                 cluster_mix_rew_start=-1, lamb=1, clusters_amount=None):
        super().__init__(k, iters, reward_class, d, user_amount, alpha,
                         cluster_initial_start, cluster_iteration_ex,
                         cluster_mix_rew_start, lamb)
        if clusters_amount is None:
            clusters_amount = [2, 3, 4]
        self.len_c = len(clusters_amount)
        self.model = Agglomerative(clusters_amount, user_amount)
