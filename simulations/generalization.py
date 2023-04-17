import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import firm_revised

seed = 100
np.random.seed(seed)
# Trying to see how sensitive computed market shares are to the initialized parameters
# All the parameters of interest 
beta = np.array([2, -0.5, -0.3])
mu = 0.5
omega = 1

# alpha = -np.exp(mu + omega**2/2)

# The number of firms in the market
J = 10

# The number of product characteristics 
K = 2

# The number of consumers in the market 
N = 500

# The number of period that I want to run the market for 
T = 100

# The product characteritics
X1 = np.random.uniform(5, 6, size=J)
X2 = np.random.normal(3, 1, size=J)
all_X = np.column_stack((X1, X2))
X0 = np.ones(J)
X = np.column_stack((X0, all_X))


# The initialized vector of prices 
max_budget = 10
p = np.zeros(J)


# Need somewhere to store the costs and the prices 
list_prices = []
list_cost = []
list_shares = []
list_markup = []
list_profit = []

# Get mean prices and costs
mean_prices = []
mean_cost = [] 


# All of this here is for the integration of the cost 
a = 0.7
b = -0.009

c_max_scale = 2.
c_min_scale = 1.

C = np.zeros((J, T))
for t in range(1, T):
    C[:, t] = b * t + a * C[:, t-1] + np.random.normal(0, 0.5, size=J)
c_max = np.tile(np.array([np.amax(C, axis=1)]).T, (1, T))
c_min = np.tile(np.array([np.amin(C, axis=1)]).T, (1, T))

C = (C - c_min) / (c_max - c_min) * (c_max_scale - c_min_scale) + c_min_scale



# Making a for loop where the cost changes each period 
for t in range (1, T): 
    v_p = np.random.normal(0, 1, size = N)
    e = np.random.gumbel(size=N*J)
    c = C[:, t]
    res1 = scipy.optimize.root(firm_revised.root_objective, p, args=(N, J, X, v_p, beta, mu, omega, e, c),
                            method='broyden2')
    optimal_price = res1.x
    for i in range(0, J):
        if optimal_price[i] < c[i]:
            optimal_price[i] = c[i]
    for x in range(0, J): 
        if optimal_price[x]> max_budget:
            optimal_price[x] = max_budget   
    shares = firm_revised.share(N, J, X, v_p, optimal_price, beta, mu, omega, e)
    profits = firm_revised.profit(optimal_price, shares, c)
    markups = firm_revised.markup(optimal_price, c)

    print(1-sum(shares))


    list_prices.append(optimal_price)
    list_cost.append(c)
    list_shares.append(shares)
    list_profit.append(profits)
    list_markup.append(markups)



# Integrate the product characteristics from the other market 
prices1 = np.array(list_prices).flatten()
cost1 = np.array(list_cost).flatten()
shares1 = np.array(list_shares).flatten()
profits1 = np.array(list_profit).flatten()
markups1 = np.array(list_markup).flatten()
products = np.tile(np.array(range(1, J+1)), T-1)
time = np.repeat(np.array(range(1, T)), J)
Car1 = np.tile(X1, T-1)
Car2 =np.tile(X2, T-1)

print(f"This is the mean markup: {np.mean(markups1)}")
print(f"This is the mean prifits: {np.mean(profits1)}")


print(prices1.shape)
print(cost1.shape)
print(time.shape)
print(Car1.shape)
print(Car2.shape)
print(shares1.shape)
print(profits1.shape)
print(markups1.shape)

df = pd.DataFrame({'price': prices1,
                   'cost': cost1,
                   'product': products, 
                   'time':time, 
                   'Car1':Car1, 
                   'Car2':Car2,
                   'mshare':shares1, 
                   'profits':profits1, 
                   'markups': markups1})
df.to_csv(f'data/market_{seed}.csv', index=False)
print(df)


