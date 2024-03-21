import math
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import pickle

# CMAB
from Strategies.RCLinUCB import RCLinUCB
from Strategies.RCLinUCBProduct import RCLinUCBProduct
# Rewards
from Strategies.Rewards.OffFeature import OffFeature

# 834596
iterations = 35000

d = 10
clusters_amounts = [1, 2, 4]

sup_percentile = 75
inf_percentile = 25

name_file = '_tuning'
with open('../datos/users' + name_file + '.pkl', 'rb') as fp:
    users = list(pickle.load(fp))

with open('../datos/articles' + name_file + '.pkl', 'rb') as fp:
    articles = list(pickle.load(fp).values())

user_amount = len(users)
k = len(articles)

reward_class = OffFeature(articles, users, name_file)
# Run experiments
cant_valores = 75
experimentos = 100
resultados = np.zeros((experimentos, iterations))
valores = np.zeros(2)
# Run experiments
for i in range(cant_valores):
    print(str(i) + "/" + str(cant_valores))
    valores[0] = random.random() * 0.9 + 0.1  # alpha
    valores[1] = random.random() * 0.9 + 0.1  # lamb
    for j in range(experimentos):
        reward_class.reset()

        linucbk = RCLinUCB(k, iterations, reward_class, d, user_amount, valores[0], clusters_amounts, valores[1], 1000,
                           500)

        resultados[j] = linucbk.run()

    linucb_inf = np.percentile(resultados, inf_percentile, axis=0)
    linucb_median_rew = np.median(resultados, axis=0)
    linucb_sup = np.percentile(resultados, sup_percentile, axis=0)

    with open('tunning_datos_' + str(i) + '.pkl', 'wb') as fp:
        pickle.dump([linucb_inf, linucb_median_rew, linucb_sup, valores], fp)
