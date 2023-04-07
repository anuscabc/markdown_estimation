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
seed = 1012
np.random.seed(seed)
# def exponeential utility: 
# This should work for normalized consumer mass to 1 

# The number of firms in the market
N = 100

# The number of product characteristics
J = 2

# put in the true param because the firm actually knows them
beta = np.array([2., -0.611, -0.528])
alpha = -2.71

average_price = []
average_profit = []
average_markup = []
outside_good_share = []

# Talk about the stochasticity when there are too large variation
all_Xj = np.random.uniform(4, 5, size=(N, J))
all_c = np.random.uniform(1, 2, size=N)

for n in range(1, N+1, 1):
    p = np.ones(n)
    # Xj = np.random.lognormal(0, 1, size = (n, J)) 
    Xj = all_Xj[:n, :]
    X0 = np.ones(n)
    X = np.column_stack((X0, Xj))
    # c = np.random.lognormal(0, 1, size = n)
    c = all_c[:n]

    # getting all the values of interest at the optimal price
    res1 = scipy.optimize.root(firm.root_objective, p, args=(n, X, beta, alpha, c),
                               method='broyden2')
    optimal_price = res1.x
    markup = firm.markup(optimal_price, c)
    share = firm.probability(optimal_price, alpha, X, beta)
    outside_good = 1 - sum(share)
    profit = firm.profit(optimal_price, share, c)

    # putting all the means in a list to append later 
    mean_optimal_price = np.mean(optimal_price)
    mean_optimal_profit = np.mean(profit)
    mean_optimal_markup = np.mean(markup)
    average_price.append(mean_optimal_price)
    average_markup.append(mean_optimal_markup)
    average_profit.append(mean_optimal_profit)
    outside_good_share.append(outside_good)

df = pd.DataFrame({'avg_price': average_price,
                   'avg_profit': average_profit,
                   'avg_markup': average_markup,
                   'outside_share': outside_good_share})
df.to_csv(f'data/sim_{seed}.csv', index=False)



