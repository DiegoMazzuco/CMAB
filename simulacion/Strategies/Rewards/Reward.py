from abc import ABC
import numpy as np


class Reward(ABC):

    def __init__(self, k: int):
        self.k = k

    def get_mu(self):
        return self.mu

    def get_products(self):
        return None

    def get_available_products(self, eventIter):
        return None

    def get_feature(self, a):
        return None

    def get_batch_size(self):
        return 0

    def get_reward(self, a):
        return self.mu[a]
