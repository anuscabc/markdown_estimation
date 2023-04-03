import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
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
# This should work for normalized consumer mass to 1 

# Trying to generalize the number of firms
N = 2

p = np.ones(N)
c = np.random.lognormal(0, 1, size = N)
X = np.random.lognormal(0, 1, size = N)


# initiate some number of consumers 


e = np.random.gumbel(0, 1, size = N)
beta = [1., 2.]
alpha = -0.5



def sum_utility(beta, X, price, alpha):
    # here think a bit about how you want to introduce the outside good 
    sum_u_exp = sum(np.exp(beta[0] + alpha*price + beta[1]*X))
    return sum_u_exp

def probability(beta, X, price, alpha, sum_u_ex): 
    q = (np.exp(beta[0] + alpha*price + beta[1]*X))/(1 + sum_u_ex)
    return q

def profit(price, beta, X,  alpha, sum_u_ex, c):
    q = probability(beta, X, price, alpha, sum_u_ex)
    Pi = - 1*q*(price - c)
    return Pi


def own_elasticites(beta, X, price, alpha, sum_u_ex):
    share = probability(beta, X, price, alpha, sum_u_ex)
    elasticity_own = alpha*share - alpha*share**2
    return elasticity_own

def cross_elasticities(beta, X, price, alpha, sum_u_ex):
    share = probability(beta, X, price, alpha, sum_u_ex)
    elasticity_cross= alpha*share**2
    return elasticity_cross



sum_q = sum_utility(beta, X, p, alpha)
prob = probability(beta, X, p, alpha, sum_q)
prof = profit(p, beta, X, alpha, sum_q, c)
own_elasticity = own_elasticites(beta, X, p, alpha, sum_q)
cross_elasticity =cross_elasticities(beta, X, p, alpha, sum_q)
# res = minimize(profit, p, args= (beta, X, alpha, sum_q, c), method = "Nelder-Mead")




print(c)
print(X)
print(sum_q)
print(prob)
print(prof)
print(own_elasticity)
print(cross_elasticity)
# print(res.x)



