import numpy as np
import pickle

from .Reward import Reward
import os

class OffFeature(Reward):

    def __init__(self, articles, users, name_file=''):
        self.articles = articles
        self.users = users
        self.users_ids = {users[i]['id']: i for i in range(len(users))}
        self.eventIter = -1
        self.event_batch = 1
        self.name_file = name_file
        with open('../datos/events1'+name_file+'.pkl', 'rb') as fp:
            self.events = pickle.load(fp)
        self.event_batch_size = len(self.events)
        self.articles_ids = {articles[i]['id']: i for i in range(len(articles))}
        files = os.listdir('../datos/')  # your directory path
        self.number_file_event = len([f for f in files if f.startswith('event')])


    def get_user(self):
        return self.users_ids[self.events[self.eventIter][3]]

    def get_products(self):
        return self.articles

    def get_batch_size(self):
        return self.event_batch_size

    def get_available_products(self):
        try:
            return [self.articles_ids[id] for id in self.events[self.eventIter][2]]
        except:
            return None

    def get_feature(self, id):
        id_user = self.get_user()
        return np.concatenate(
            [self.articles[id]['features'], np.array(self.users['id' == id_user]['features']).reshape(5, 1)])

    def reset(self):
        self.eventIter = -1
        self.event_batch = 1
        with open('../datos/events1'+self.name_file+'.pkl', 'rb') as fp:
            self.events = pickle.load(fp)

    def advance_iter(self):
        self.eventIter += 1
        if self.eventIter == self.event_batch_size:
            self.eventIter = 0
            self.event_batch += 1
            if self.event_batch >= self.number_file_event:
                return False
            with open('../datos/events' + str(self.event_batch) + self.name_file +'.pkl', 'rb') as fp:
                self.events = pickle.load(fp)
            self.event_batch_size = len(self.events)
        if len(self.events) == self.eventIter:
            return False
        else:
            return True

    def get_reward(self, user_id: int, a: int):
        if self.events[self.eventIter][0] == self.articles[a]['id']:
            if self.events[self.eventIter][1]:
                return 1
            else:
                return 0
        else:
            return None
