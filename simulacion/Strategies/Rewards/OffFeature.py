import numpy as np
import pickle

from .Reward import Reward


class OffFeature(Reward):

    def __init__(self, k: int, d: int, articles):
        super().__init__(k, d)
        self.articles = articles
        self.eventIter = -1
        self.event_batch = 1
        with open('../datosReales/yahoo/events1.pkl', 'rb') as fp:
            self.events = pickle.load(fp)
        self.event_batch_size = len(self.events)

    def get_products(self):
        return self.articles

    def get_batch_size(self):
        return self.event_batch_size

    def get_available_products(self, eventIter):
        try:
            return self.events[eventIter][2]
        except:
            return None

    def get_feature(self, id):
        return np.array(self.articles[id]['features'])

    def reset(self):
        self.eventIter = -1

    def get_reward(self, a):
        self.eventIter += 1
        if self.eventIter == self.event_batch_size:
            self.eventIter = 0
            self.event_batch += 1
            with open('../datosReales/yahoo/events'+str(self.event_batch)+'.pkl', 'rb') as fp:
                self.events = pickle.load(fp)
            self.event_batch_size = len(self.events)
        if len(self.events) == self.eventIter:
            return None
        if self.events[self.eventIter][0] == self.articles[a]['id']:
            if self.events[self.eventIter][1]:
                return 1
            else:
                return 0
        else:
            return None
