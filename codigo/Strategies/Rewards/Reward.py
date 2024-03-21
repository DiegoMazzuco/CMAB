from abc import ABC
import numpy as np


class Reward(ABC):

    def get_products(self):
        return None

    def get_available_products(self):
        return None

    def get_feature(self, a):
        return None

    def get_batch_size(self):
        return 0

    def get_reward(self, user_id: int, a: int):
        return None

    def advance_iter(self):
        return True

    def reset(self):
        pass
