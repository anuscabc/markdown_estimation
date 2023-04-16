import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import firm_revised

seed = 50
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
T = 10

# The product characteritics
X1 = np.random.uniform(3, 8, size=J)
X2 = np.random.normal(3, 1, size=J)
all_X = np.column_stack((X1, X2))
X0 = np.ones(J)
X = np.column_stack((X0, all_X))


# The initialized vector of prices 
p = np.ones(J)


# Need somewhere to store the costs and the prices 
list_prices = []
list_cost = []



X1 = np.random.uniform(3, 4, size=J)
X2 = np.random.normal(3, 1, size=J)
all_X = np.column_stack((X1, X2))
X0 = np.ones(J)
X = np.column_stack((X0, all_X))


# All of this here is for the integration of the cost 
a = 0.8
b = -0.0009

c_max_scale = 4.
c_min_scale = 1.

C = np.zeros((J, T))
for t in range(1, T):
    C[:, t] = b * t + a * C[:, t-1] + np.random.normal(0, 1, size=J)
c_max = np.tile(np.array([np.amax(C, axis=1)]).T, (1, T))
c_min = np.tile(np.array([np.amin(C, axis=1)]).T, (1, T))

C = (C - c_min) / (c_max - c_min) * (c_max_scale - c_min_scale) + c_min_scale



# Making a for loop where the cost changes each period 
for t in range (1, T): 
    v_p = np.random.normal(0, 1, size = N)
    e = np.random.gumbel(size=N*J)
    c = C[:, t]
    res1 = scipy.optimize.root(firm_revised.root_objective, p, 
                               args=(N, J, X, v_p, beta, mu, omega, e, c), method='broyden2')
    optimal_price = res1.x
    list_prices.append(optimal_price[1])
    list_cost.append(c[1])


# Integrate the product characteristics from the other market 
print(list_prices)
print(list_cost)



# df = pd.DataFrame({'price': list_prices,
#                    'cost': list_cost})
# df.to_csv(f'data/individual_{seed}.csv', index=False)


