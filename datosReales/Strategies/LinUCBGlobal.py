import numpy as np

from .MAB import MAB
from .Rewards import Reward


class LinUCBGlobal(MAB):
    def __init__(self, k: int, iters: int, reward_class: Reward, d: int, user_amount: int, alpha: float):
        super().__init__(k, iters, reward_class)
        self.d = d

        self.A = np.identity(d)
        self.b = np.zeros([d, 1])
        self.alpha = alpha
        self.eventIter = 0
        self.users = reward_class.get_users().keys()

    def get_user_indice(self, id):
        return self.users.index(id)

    def calc_ucb(self, product_id, user_id):
        #Se ignora user_id pq se supone todos los usuarios se comportan igual en un mismo cluster

        x = self.reward_class.get_feature(product_id).reshape((self.d, 1))
        A_inv = np.linalg.inv(self.A)
        theta = np.dot(A_inv, self.b)
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p[0]

    def reward_update(self, reward, id, user_id):
        #Se ignora user_id pq se supone todos los usuarios se comportan igual en un mismo cluster
        x = self.reward_class.get_feature(id).reshape((self.d, 1))
        self.A += np.dot(x, x.T)
        self.b += reward * x

    def select_action(self, user_id):
        bandits = np.zeros(self.k)

        available_products = self.reward_class.get_available_products(self.eventIter)
        if available_products is not None:
            # Algunos productos nunca son recomendados entonces hay q tomar intersection
            available_products = np.intersect1d(self.articles, available_products)
            for i in range(len(available_products)):
                id = available_products[i]
                bandits[i] = self.calc_ucb(id, user_id)

            return available_products[np.argmax(bandits)]
        else:
            return None

    def pull_reward(self):

        obtuvo_valido = False
        reward = None
        id = None
        while not obtuvo_valido:

            user_id = self.reward_class.get_corresponding_user(self.eventIter)

            id = self.select_action(user_id)

            reward = None
            if id is not None:
                reward = self.reward_class.get_reward(id)

            if reward is not None:
                obtuvo_valido = True
            else:
                obtuvo_valido = False

            self.eventIter += 1
            if self.eventIter >= self.reward_class.get_batch_size():
                self.eventIter = 0

        self.reward_update(reward, id, user_id)

        return reward
