import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, jacobian
# importing the data simulation functions 
# import demand_data_simulation
# import clean_data
# import demand_data_estimation
import statsmodels.api as sm
import statsmodels.formula.api as smf# from main import beta, mu, omega, df

# This can be the share data as defined in the simulation part i think 
np.random.seed(4)
# def exponeential utility: 
# This should work for normalized consumer mass to 1 

# Trying to generalize the number of firms
N = 1
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

def probability(beta, X, price, alpha): 
    q = (np.exp(beta[0] + alpha*price + beta[1]*X))/(1 + sum(np.exp(beta[0] + alpha*price + beta[1]*X)))
    return q

def profit(price, beta, X,  alpha, c):
    q = probability(beta, X, price, alpha)
    Pi = - 1*q*(price - c)
    return Pi

# def own_elasticites(beta, X, price, alpha):
#     share = probability(beta, X, price, alpha)
#     elasticity_own = alpha*share - alpha*share**2
#     return elasticity_own

# def cross_elasticities(beta, X, price, alpha):
#     share = probability(beta, X, price, alpha)
#     elasticity_cross= -alpha*share*share
#     return elasticity_cross

# Lets see if possible to get the same Jacobian like this as 

def derivative(N, beta, X, price, alpha):
    share = probability(beta, X, price, alpha)
    J = np.zeros((N, N)) 
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if i == j: 
                J[i, j] = alpha*share[i] - alpha*share[i]**2
            if i!= j: 
                J[i, j] = -alpha*share[i]*share[j]
    return J



def root_objective(price, N, beta, X, alpha):
    share = probability(beta, X, price, alpha)
    J = derivative(N, beta, X, price, alpha) 
    optim = np.matmul(np.transpose(J), (price - c)) + share
    return optim


# res = minimize(profit, p, args= (beta, X, alpha, c), method = "Nelder-Mead")
# print(res.x)


# initial optimization with 2 firms 

res1 = scipy.optimize.root(root_objective, p, args=(N, beta, X, alpha), method = 'hybr')

optimal_price =


deriv = derivative(N, beta,X, p, alpha)
sum_q = sum_utility(beta, X, p, alpha)
prof = profit(p, beta, X, alpha, c)

