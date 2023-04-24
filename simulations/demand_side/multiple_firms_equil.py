import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
# import scipy.stats
import matplotlib.pyplot as plt
# importing the data simulation functions 
# import demand_data_simulation
# import clean_data
# import demand_data_estimation
import firm_revised

# This can be the share data as defined in the simulation part i think 
seed = 120
np.random.seed(seed)
# def exponeential utility: 
# This should work for normalized consumer mass to 1 


# Number of consumers 
N = 500


# The number of firms in the market
J = 10

# The number of product characteristics
K = 2

# put in the true param because the firm actually knows them
beta = np.array([2, -0.5, -0.3])
mu = 0.5
omega = 1

average_price = []
average_profit = []
average_markup = []
outside_good_share = []
average_cost = []

X1 = np.random.uniform(5, 6, size=J)
X2 = np.random.uniform(5, 6, size=J)
all_Xj = np.column_stack((X1, X2))
all_c = np.random.uniform(3, 3, size = J)
v_p = np.random.normal(0, 1, N)

for j in range(1, J+1, 1):
    p = np.ones(j)
    Xj = all_Xj[:j, :]
    X0 = np.ones(j)
    X = np.column_stack((X0, Xj))
    c = all_c[:j]
    e = 0.


    # getting all the values of interest at the optimal price
    res1 = scipy.optimize.root(firm_revised.root_objective, p, args=(N, j, X, v_p, beta, mu, omega, e, c),
                               method='broyden2')
    optimal_price = res1.x
    print(optimal_price)
    print(c)
    markup = firm_revised.markup(optimal_price, c)
    share = firm_revised.share(N, j, X, v_p, optimal_price, beta, mu, omega, e)[0]
    outside_good = 1 - sum(share)
    print(outside_good)
    profit = firm_revised.profit(optimal_price, share, c)

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



