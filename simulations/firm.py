import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt






def sum_utility(beta, X, price, alpha):
    sum_u_exp = sum(np.exp(beta[0] + alpha*price + beta[1]*X))
    return sum_u_exp

def probability(price, alpha, X, beta): 
    q = (np.exp(X@beta + alpha*price))/(1 + sum(np.exp(X@beta + alpha*price)))
    return q

def profit(price, beta, X,  alpha, c):
    q = probability(beta, X, price, alpha)
    Pi = q*(price - c)
    return Pi

def markup(price, c): 
    markup = (price - c)/c
    return markup


def own_elasticites(beta, X, price, alpha):
    share = probability(beta, X, price, alpha)
    elasticity_own = alpha*share - alpha*share**2
    return elasticity_own

def cross_elasticities(beta, X, price, alpha):
    share = probability(beta, X, price, alpha)
    elasticity_cross= -alpha*share*share
    return elasticity_cross


def derivative(N, alpha, share):
    J = np.zeros((N, N))
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if i == j: 
                J[i, j] = alpha*share[i] - alpha*share[i]**2
            elif i!= j: 
                J[i, j] = -alpha*share[i]*share[j]
    return J

def root_objective(price, N, X, beta, alpha, c):
    share = (np.exp(X@beta + alpha*price))/(1 + sum(np.exp(X@beta + alpha*price)))
    J = derivative(N, alpha, share) 
    optim = np.matmul(np.transpose(J), (price - c)) + share
    return optim




