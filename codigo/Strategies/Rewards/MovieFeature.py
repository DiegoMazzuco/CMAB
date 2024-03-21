import numpy as np
import pickle

from .Reward import Reward
import os

class MovieFeature(Reward):

    def __init__(self, articles):
        self.articles = articles
        self.eventIter = -1
        self.event_batch = 1
        with open('../datos/movielens/events1_movie.pkl', 'rb') as fp:
            self.events = pickle.load(fp)
        self.event_batch_size = len(self.events)
        files = os.listdir('../datos/')
        self.number_file_event = len([f for f in files if f.startswith('event')])


    def get_user(self):
        return self.events[self.eventIter][2]

    def get_products(self):
        return self.articles

    def get_batch_size(self):
        return self.event_batch_size

    def get_available_products(self):
        try:
            return self.events[self.eventIter][1]
        except:
            return None

    def get_feature(self, id):
        return self.articles[id]

    def reset(self):
        self.eventIter = -1
        self.event_batch = 1
        with open('../datos/movielens/events1_movie.pkl', 'rb') as fp:
            self.events = pickle.load(fp)

    def advance_iter(self):
        self.eventIter += 1
        if self.eventIter == self.event_batch_size:
            self.eventIter = 0
            self.event_batch += 1
            if self.event_batch >= self.number_file_event:
                return False
            with open('../datos/events' + str(self.event_batch) + '_movie.pkl', 'rb') as fp:
                self.events = pickle.load(fp)
            self.event_batch_size = len(self.events)
        if len(self.events) == self.eventIter:
            return False
        else:
            return True

    def get_reward(self, user_id: int, a: int):
        if self.events[self.eventIter][0] == a:
            return 1
        else:
            return 0
