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
np.random.seed(7)
# def exponeential utility: 
# This should work for normalized consumer mass to 1 

# The number of firms in the market
N = 10

# The number of product characteristics
J = 2

# put in the true param because the firm actually knows them
beta = np.array([2., -0.611, -0.528])
alpha = -2.71

average_price = []
# average_profit = []
# average_markup = []

for n in range(1, N+1, 1):
    p = np.ones(n)
    Xj = np.random.lognormal(0, 1, size = (n, J)) 
    X0 = np.ones(n)
    X = np.column_stack((X0, Xj))
    c = np.random.lognormal(0, 1, size = n)
    # getting all the values of interest at the optimal price
    res1 = scipy.optimize.root(firm.root_objective, p, args=(n, X, beta, alpha, c), method = 'broyden2')
    optimal_price = res1.x
    # markup = firm.markup(optimal_price, c)
    # share = firm.probability(optimal_price, alpha, X, beta)
    # profit = firm.profit(optimal_price, share, c)
    # putting all the means in a list to append later 
    mean_optimal_price = np.mean(optimal_price)
    # mean_optimal_profit = np.mean(profit)
    # mean_optimal_markup = np.mean(markup)
    average_price.append(mean_optimal_price)
    # average_markup.append(mean_optimal_markup)
    # average_profit.append(mean_optimal_profit)

print(average_price)
# print(average_profit)
# print(average_markup)

x_axis = range(1, N+1, 1)

# fig, axs = plt.subplots(3)
# fig.suptitle('Market Equilibium Over Different Number of Frims')
# axs[0].plot(x_axis, average_price, 'tab:orange')
# axs[0].set_title("Price")
# axs[1].plot(x_axis, average_profit, 'tab:green')
# axs[1].set_title("Profit")
# axs[2].plot(x_axis, average_markup)
# axs[2].set_title("Markup")
# fig.tight_layout()
# fig.savefig("plots/all.png")

