import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
# importing the data simulation functions 
# import demand_data_simulation
# import clean_data
# import demand_data_estimation
import firm

# This can be the share data as defined in the simulation part i think 
seed = 50
np.random.seed(seed)
# def exponeential utility: 
# This should work for normalized consumer mass to 1 

# The number of firms in the market
N = 10
# The number of product characteristics
J = 2

# put in the true param because the firm actually knows them
beta = np.array([2, 0.611, -0.528])
mu = 0.5
omega = 1

#the mean alpha is 
alpha = - np.exp(mu + omega**2/2)

average_price = []
average_profit = []
average_markup = []
outside_good_share = []
average_cost = []

# Talk about the stochasticity when there are too large variation
e = np.random.normal(size = N)
all_Xj = np.random.uniform(4.,5., size=(N, J))
all_c = np.random.uniform(3, 3, size = N)

# need to somehow include the errors in the estimation 
# need to sit down and make some sort of programming scheme and rewrite a bunch of stuff 
# also need to check where some of the stuff goes wrong 

for n in range(1, N+1, 1):
    p = np.ones(n)
    Xj = all_Xj[:n, :]
    X0 = np.ones(n)
    X = np.column_stack((X0, Xj))
    c = all_c[:n]

    # getting all the values of interest at the optimal price
    res1 = scipy.optimize.root(firm.root_objective, p, args=(n, X, beta, alpha, c),
                               method='broyden2')
    optimal_price = res1.x
    print(optimal_price)
    for i in range(0, n): 
        if optimal_price[i] < c[i]:
            optimal_price[i]=c[i]
    print(optimal_price)
    print(c)
    markup = firm.markup(optimal_price, c)
    share = firm.probability(optimal_price, alpha, X, beta)
    outside_good = 1 - sum(share)
    print(outside_good)
    profit = firm.profit(optimal_price, share, c)

    # putting all the means in a list to append later 
    mean_optimal_price = np.mean(optimal_price)
    mean_optimal_profit = np.mean(profit)
    mean_optimal_markup = np.mean(markup)
    mean_cost = np.mean(c)
    average_price.append(mean_optimal_price)
    average_markup.append(mean_optimal_markup)
    average_profit.append(mean_optimal_profit)
    outside_good_share.append(outside_good)
    average_cost.append(mean_cost)

df = pd.DataFrame({'avg_price': average_price,
                   'avg_profit': average_profit,
                   'avg_markup': average_markup,
                   'outside_share': outside_good_share,
                   'cost': average_cost})
df.to_csv(f'data/sim_{seed}.csv', index=False)



