import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import optimize
# import scipy.optimize
import matplotlib.pyplot as plt
# importing the data simulation functions 
# import demand_data_simulation
# import clean_data
# import demand_data_estimation
import statsmodels.api as sm
import statsmodels.formula.api as smf
# from main import beta, mu, omega, df




# This can be the share data as defined in the simulation part i think 
np.random.seed(4)
# def exponeential utility: 
p = np.array([1., 1., 0])
X = np.array([0.3, 0.4, 0])
beta = [1, 2]
alpha = -0.5
# initiate some number of consumers 
N = 1
e = np.random.gumbel(0, 1, size = len(p*N))

def utility_exponetial(beta, X, price, alpha, e):
    u_exp = np.exp(beta[0] + alpha*price + beta[1]*X+ e)
    return u_exp

def sum_utility(beta, X, price, alpha, e):
    # here think a bit about how you want to introduce the outside good 
    sum_u_exp = sum((np.exp(beta[0] + alpha*price + beta[1]*X+ e)[:-1]))
    return sum_u_exp

def quantity(beta, X, price, alpha, sum_u_ex, e): 
    q = (np.exp(beta[0] + alpha*price + beta[1]*X+ e))/(1 + sum_u_ex)
    return q

def quantity_more_consumers(beta, X, price, alpha, sum_u_ex, e): 
    for i in range(1, N+1, 1):
        q = (np.exp(beta[0] + alpha*price + beta[1]*X+ e))/(1 + sum_u_ex)
    return q


utilities = utility_exponetial(beta, X, p, alpha, e)
sum = sum_utility(beta, X, p, alpha, e)
quantity_check = quantity(beta, X, p, alpha, sum, e)

print(e)
print(utilities)
print(sum)
print(quantity_check)
# def share(price_own, price, beta, alpha, x_2_own, x_2, x_3_own, x_3):

#     return share


# def profit(x_own, x_rest, c_own, b):
#     return share(x_own, x_rest, b) * (x_own - c_own)

# def neg_profit(x_own, x_rest, c_own, b):
#     return -1 * profit(x_own, x_rest, c_own, b)


# def reaction(x, c, b, i):
#     c_own = c[i]
#     x_rest = np.delete(x, i)
#     params = (x_rest, c_own, b)
#     x_opt = optimize.brute(neg_profit, ((0, 1,),), args=params)
#     return x_opt[0]


# def vector_reaction(x, params): 
#     b, c, n_firms = params
#     return np.array(x) - np.array([reaction(x, c, b, i) for i in range(n_firms)])


# np.random.seed(1012)


# b0 = 5.
# for n_firms in range(1, 10, 1):
#     c = np.random.uniform(1,4,n_firms)
#     params = [b0, c, n_firms]
#     x0 = np.ones(n_firms)
#     ans = optimize.fsolve(vector_reaction, x0, args=(params))
#     print(ans)


# try to get the fixed point iteration to somewhat work 

