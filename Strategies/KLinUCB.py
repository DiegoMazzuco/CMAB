import numpy as np

from .CLinUCB import CLinUCB
from .Clusters import Kmeans
from .MAB import MAB
from .Rewards import BernoulliFeature


class KLinUCB(CLinUCB):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, d: int, user_amount: int, alpha: float,
                cluster_initial_start = 2000, cluster_iteration_ex=1000,
                 cluster_mix_rew_start=-1, lamb=1, cluster_amount=2, cluster_it=50):
        super().__init__(k, iters, reward_class, d, user_amount, alpha,
                         cluster_initial_start, cluster_iteration_ex,
                         cluster_mix_rew_start, lamb)
        self.model = Kmeans(cluster_amount, cluster_it, user_amount)
