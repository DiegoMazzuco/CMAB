# Tests
import cProfile
from pstats import Stats
import os
import numpy as np

from simulacion.Strategies.RCLinUCB import RCLinUCB
from simulacion.Strategies.Rewards.BernoulliFeature import BernoulliFeature

def main():
    iterations = 10000
    k = 100
    d = 10
    alpha = 1
    user_amount = 6
    max_prob = 0.3
    noise = 0.05
    aux1 = np.zeros(d)
    aux1[0] = 1
    aux2 = np.zeros(d)
    aux2[1] = 1
    best_theta = [aux1, aux2]
    lamb = 1
    clusters_amounts = [1, 2, 4]

    reward_class = BernoulliFeature(k, d, user_amount, max_prob, noise, best_theta)
    linucbg     = RCLinUCB(k, iterations, reward_class, d, user_amount, alpha, clusters_amounts, lamb, 500, 500 )
    linucbg.run()

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(10)