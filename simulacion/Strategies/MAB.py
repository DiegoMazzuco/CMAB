from abc import ABC

import random as rd
import numpy as np

from simulacion.Strategies.Rewards.BernoulliFeature import BernoulliFeature


class MAB(ABC):
    def __init__(self, k: int, iters: int, reward_class: BernoulliFeature, user_amount: int):
        """
            Sum up two integers
            Arguments:
                k: Number of arms
                iters: Number of iterations
                reward: Class to get the reward
        """
        # Number of arms
        self.k = k
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Number of iterations
        self.iters = iters
        # Total mean reward
        self.mean_reward = 0
        self.rewards = np.zeros(iters)
        # Reward Class
        self.reward_class = reward_class
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        #
        self.user_amount = user_amount

    def calc_ucb(self, i: int, user_id: int):
        """
            Ver implementaciones
        """
        return None

    def reward_update(self, reward, i, user_id):
        """
            Ver implementaciones
        """
        return None

    def select_action(self, user_id: int):
        bandits = np.zeros(self.k)

        for i in range(self.k):
            bandits[i] = self.calc_ucb(i, user_id)

        return np.argmax(bandits)

    def pull_reward(self, user_id):
        a = self.select_action(user_id)

        reward = self.reward_class.get_reward(user_id, a)

        if reward is None:
            return None
        self.reward_update(reward, a, user_id)

        return reward

    def run(self):
        """
            Run a simulation Returning a vector with the mean reward in each iteration
            Returns:
                mean rewards
        """
        for i in range(self.iters):
            # Se elije un usuario al azar
            user_id = rd.randint(0, self.user_amount-1)
            reward = self.pull_reward(user_id)
            if reward is None:
                break
            # Update the average
            self.mean_reward = self.mean_reward + (
                    reward - self.mean_reward) / (i + 1)
            # Save the average
            self.rewards[i] = self.mean_reward
            if i % 1000 == 0:
                print(str(i)+'/'+str(self.iters))
        return self.rewards
