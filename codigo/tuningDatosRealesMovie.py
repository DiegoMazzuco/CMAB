import random
import numpy as np
import pickle

# CMAB
from Strategies.RCLinUCB import RCLinUCB
# Rewards
from codigo.Strategies.Rewards.MovieFeature import MovieFeature

# 834596
iterations = 35000

d = 25
clusters_amounts = [1, 2, 4]

sup_percentile = 75
inf_percentile = 25

with open('../datos/movielens/articles_movie.pkl', 'rb') as fp:
    articles = list(pickle.load(fp).values())

user_amount = 117
k = len(articles)

reward_class = MovieFeature(articles)
# Run experiments
cant_valores = 75
experimentos = 1000
resultados = np.zeros((experimentos, iterations))
valores = np.zeros(2)
# Run experiments
for i in range(cant_valores):
    print(str(i) + "/" + str(cant_valores))
    valores[0] = random.random() * 0.9 + 0.1  # alpha
    valores[1] = random.random() * 0.9 + 0.1  # lamb
    for j in range(experimentos):
        print(str(j) + "/" + str(experimentos))
        reward_class.reset()

        linucbk = RCLinUCB(k, iterations, reward_class, d, user_amount, valores[0], clusters_amounts, valores[1], 1000,
                           500)

        resultados[j] = linucbk.run()

    linucb_inf = np.percentile(resultados, inf_percentile, axis=0)
    linucb_median_rew = np.median(resultados, axis=0)
    linucb_sup = np.percentile(resultados, sup_percentile, axis=0)

    with open('movie_tunning_datos_' + str(i) + '.pkl', 'wb') as fp:
        pickle.dump([linucb_inf, linucb_median_rew, linucb_sup, valores], fp)
