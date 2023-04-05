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

# define the global parameter and the global number of firms 
N = 20

# put in the true param because the firm actually knows them
beta = [2., -0.611]
alpha = -2.71


average_price = []
average_profit = []
average_markup = []


for n in range(1, N+1, 1):
    p = np.ones(n)
    c = np.random.lognormal(0, 1, size = n)
    X = np.random.lognormal(0, 1, size = n)
    e = np.random.gumbel(0, 1, size = n)

    # getting all the values of interest at the optimal price
    res1 = scipy.optimize.root(firm.root_objective, p, args=(n, beta, X, alpha, c), method = 'broyden2')
    optimal_price = res1.x
    profit = firm.profit(optimal_price, beta, X, alpha, c)
    markup = firm.markup(optimal_price, c)

    # putting all the means in a list to append later 
    mean_optimal_price = np.mean(optimal_price)
    mean_optimal_profit = np.mean(profit)
    mean_optimal_markup = np.mean(markup)
    average_price.append(mean_optimal_price)
    average_profit.append(mean_optimal_profit)
    average_markup.append(mean_optimal_markup)

print(average_price)
print(average_profit)
print(average_markup)

x_axis = range(1, N+1, 1)
# plt.plot(x_axis, average_price)
# plt.title('Average market price equilibirum')
# plt.xlabel('Number of firms')
# plt.ylabel('Price')
# plt.savefig('plots/average_price.png')


# plot_profit =  plt.plot(x_axis, average_profit)
# plt.title('Average market profit equilibirum')
# plt.xlabel('Number of firms')
# plt.ylabel('Price')
# plt.savefig('plots/average_profit.png')


# plot_mark_up =  plt.plot(x_axis, average_mark_up)
# plt.title('Average market markup equilibirum')
# plt.xlabel('Number of firms')
# plt.ylabel('Price')
# plt.savefig('plots/average_markup.png')

fig, axs = plt.subplots(3)
fig.suptitle('Market Equilibium Over Different Number of Frims')
axs[0].plot(x_axis, average_price, 'tab:orange')
axs[0].set_title("Price")
axs[1].plot(x_axis, average_profit, 'tab:green')
axs[1].set_title("Profit")
axs[2].plot(x_axis, average_markup)
axs[2].set_title("Markup")
fig.tight_layout()
fig.savefig("plots/all.png")

