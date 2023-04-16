import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import firm_revised

np.random.seed(100)
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

# Schockes per consumer 
v_p = np.random.normal(0, 1, size = N)

X1 = np.random.uniform(3, 4, size=J)
X2 = np.random.normal(3, 1, size=J)
all_X = np.column_stack((X1, X2))
X0 = np.ones(J)
X = np.column_stack((X0, all_X))

p = np.ones(J)
e = np.random.gumbel(size=N*J)
c = np.random.uniform(1, 2, size=J)



res1 = scipy.optimize.root(firm_revised.root_objective, p, args=(N, J, X, v_p, beta, mu, omega, e, c),
                            method='broyden2')
optimal_price = res1.x
for i in range(0, J):
    if optimal_price[i] < c[i]:
        optimal_price = c

optimal_shares = firm_revised.share(N, J, X, v_p, optimal_price, beta, mu, omega, e)
markups = firm_revised.markup(optimal_price, c)
profits = firm_revised.profit(optimal_price, optimal_shares, c)
print(optimal_price)
print(c)
print(optimal_shares)
print(profits)


# 