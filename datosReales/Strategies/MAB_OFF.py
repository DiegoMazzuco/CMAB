from abc import ABC
import numpy as np

from .Rewards import Reward


class MAB_OFF(ABC):
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
        self.articles = reward_class.get_products().keys()
        self.articles = [article for article in self.articles]
        self.eventIter = -1
        #Esto son valores estimados para cada brazo, osea la media estimada o el ucb.
        self.values = {}
        for i in range(k):
            id = self.articles[i]
            self.values[id] = self.calc_value(id)

    def get_article_index(self, id):
        return self.articles.index(id)

    def select_action(self):
        available_products = self.reward_class.get_available_products(self.eventIter)
        bandits = np.zeros(len(available_products))
        if available_products is not None:
            #Algunos productos nunca son recomendados entonces hay q tomar intersection
            available_products = np.intersect1d(self.articles, available_products)
            for i in range(len(available_products)):
                id = available_products[i]
                bandits[i] = self.values[id]

            return available_products[np.argmax(bandits)]
        else:
            return None

    def calc_value(self, id):
        return None

    def reward_update(self, reward, id):
        i = self.get_article_index(id)
        # Update counts
        self.n += 1
        self.k_n[i] += 1

        # Update results for a_k
        self.k_reward[i] = self.k_reward[i] + (
                reward - self.k_reward[i]) / self.k_n[i]
        self.values[id] = self.calc_value(id)

    def pull_reward(self):

        obtuvo_valido = False
        reward = None
        id = None
        while not obtuvo_valido:
            self.eventIter += 1
            if self.eventIter >= self.reward_class.get_batch_size():
                self.eventIter = 0

            id = self.select_action()

            reward = None
            if id is not None:
                reward = self.reward_class.get_reward(id)

            if reward is not None:
                obtuvo_valido = True
            else:
                obtuvo_valido = False

        self.reward_update(reward, id)

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
