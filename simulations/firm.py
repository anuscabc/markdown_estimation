import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import optimize
# import scipy.optimize
import matplotlib.pyplot as plt
# importing the data simulation functions 
import demand_data_simulation
import clean_data
import demand_data_estimation
import statsmodels.api as sm
import statsmodels.formula.api as smf
from main import X, beta, mu, omega 


# class Firm: 
#     def __init__(self, id, shares, cost):
        
#         self.id = id
#         self.shares = shares
#         self.cost = cost 

# make it work for 1 time period
X = clean_data.clear_outside_good(X)
alpha = -np.exp((mu + omega**2/2))
print(X)

def demand(price_own, price, beta, alpha, x_2_own, x_2, x_3_own, x_3):
    share = (np.exp(beta[0] + beta[1]*x_2_own + beta[3]*x_3_own + alpha*price_own))/(1 + sum(np.exp(beta[0] + beta[1]*x_2 + beta[3]*x_3 + alpha*price)))
    return share

# Check how to pass the arguments to the price thing: 

price.np.random

sum = sum(np.exp(beta[0] + beta[1]* X["x_2"] + beta[3]*X["x_3"] + alpha*price)))


# The price changes across the timeline will come from the variation on the cost 
# this needs to be implemented further 
def profit(x_own, x_rest, c_own, b):
    return demand(x_own, x_rest, b) * (x_own - c_own)

def neg_profit(x_own, x_rest, c_own, b):
    return -1 * profit(x_own, x_rest, c_own, b)


def reaction(x, c, b, i):
    c_own = c[i]
    x_rest = np.delete(x, i)
    params = (x_rest, c_own, b)
    x_opt = optimize.brute(neg_profit, ((0, 1,),), args=params)
    return x_opt[0]


def vector_reaction(x, params): 
    b, c, n_firms = params
    return np.array(x) - np.array([reaction(x, c, b, i) for i in range(n_firms)])


np.random.seed(1012)


b0 = 5.
for n_firms in range(1, 10, 1):
    c = np.random.uniform(1,4,n_firms)
    params = [b0, c, n_firms]
    x0 = np.ones(n_firms)
    ans = optimize.fsolve(vector_reaction, x0, args=(params))
    print(ans)

