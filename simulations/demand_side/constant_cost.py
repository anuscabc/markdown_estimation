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
omega = 1.

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
X1 = np.random.uniform(1, 50, size=J)
X2 = X1
# X2 = np.random.uniform(1, 10, size=J)
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
mean_vp = []



C = np.zeros((J, T))+ 3



# Making a for loop where the cost changes each period 
for t in range (1, T): 
    # v_p = np.zeros(N)
    e = 0.
    c = C[:, t]
    v_p = np.random.normal(0, 1, size=N)


    res1 = scipy.optimize.root(
        firm_revised.root_objective, 
        p, 
        args=(N, J, X, v_p, beta, mu, omega, e, c),
        method='broyden2'
    )
    optimal_price = res1.x

    shares_per_firm, all_probs = firm_revised.share(N, J, X, v_p, optimal_price, beta, mu, omega, e)
    profits = firm_revised.profit(optimal_price, shares_per_firm, c)

    markups = firm_revised.markup(optimal_price, c)

    list_prices.append(optimal_price)
    list_cost.append(c)
    list_shares.append(shares_per_firm)
    list_profit.append(profits)
    list_markup.append(markups)
    mean_vp.append(np.mean(v_p))


# Integrate the product characteristics from the other market 
prices1 = np.array(list_prices).flatten()
cost1 = np.array(list_cost).flatten()
shares1 = np.array(list_shares).flatten()
profits1 = np.array(list_profit).flatten()
markups1 = np.array(list_markup).flatten()
products = np.tile(np.array(range(1, J+1)), T-1)
time = np.repeat(np.array(range(1, T)), J)
Car1 = np.tile(X1, T-1)
Car2 = np.tile(X2, T-1)
shocks = np.repeat(np.array(mean_vp), J)

print(shocks.shape)

print(f"This is the mean markup: {np.mean(markups1)}")
print(f"This is the mean profits: {np.mean(profits1)}")




df = pd.DataFrame({'price': prices1,
                   'cost': cost1,
                   'product': products, 
                   'time':time, 
                   'Car1':Car1, 
                   'Car2':Car2,
                   'mshare':shares1, 
                   'profits':profits1, 
                   'markups': markups1, 
                   'shocks': shocks
                   })
df.to_csv(f'data/market_constant_cost{seed}.csv', index=False)
print(df)
