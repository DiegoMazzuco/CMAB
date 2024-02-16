from abc import ABC
import numpy as np

from .Rewards import Reward


class MAB(ABC):
    def __init__(self, k: int, iters: int, reward_class: Reward):
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

    def select_action(self, user_id):
        """
            Take the best action and update the internals parameters
        """
        return self.k

    def pull_reward(self):
        a = self.select_action()

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        reward = self.reward_class.get_reward(a)

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
                reward - self.k_reward[a]) / self.k_n[a]

        return reward

    def run(self):
        """
            Run a simulation Returning a vector with the mean reward in each iteration
            Returns:
                mean rewards
        """
        for i in range(self.iters):
            reward = self.pull_reward()
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
