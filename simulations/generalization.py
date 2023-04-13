import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import firm

seed = 9
np.random.seed(seed)
# The number of firms in the market
N = 10

# The number of product characteristics
caract = 2

# The number of time periods in the model 
T = 100
# The number of markets

# put in the true param because the firm actually knows them
beta = np.array([2., -0.611, -0.528])
alpha = -2.71


p = np.ones(N)
Xj = np.random.lognormal(0, 1, size = (N, caract)) 
X0 = np.ones(N)
X = np.column_stack((X0, Xj))


list_prices = []
list_cost = []


a = 0.9
b = -0.0009


c_max_scale = 4.
c_min_scale = 1.

C = np.zeros((N, T))
for t in range(1, T):
    C[:, t] = b * t + a * C[:, t-1] + np.random.normal(0, 1, size=N)
c_max = np.tile(np.array([np.amax(C, axis=1)]).T, (1, T))
c_min = np.tile(np.array([np.amin(C, axis=1)]).T, (1, T))

C = (C - c_min) / (c_max - c_min) * (c_max_scale - c_min_scale) + c_min_scale

# Making a for loop where the cost changes each period 
for t in range (1, T): 
    # c = np.random.lognormal(0, 1, size = N)
    c = C[:, t]
    res1 = scipy.optimize.root(firm.root_objective, p, 
                               args=(N, X, beta, alpha, c), method='broyden2')
    optimal_price = res1.x
    list_prices.append(optimal_price[1])
    list_cost.append(c[1])

# Make it in array form 
# array_prices = np.concatenate(list_prices).ravel()
# array_cost = np.concatenate(list_cost).ravel()


# Integrate the product characteristics from the other market 
print(list_prices)
print(list_cost)



df = pd.DataFrame({'price': list_prices,
                   'cost': list_cost})
df.to_csv(f'data/individual_{seed}.csv', index=False)


