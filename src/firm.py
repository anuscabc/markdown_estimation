from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


# class Firm: 
#     def __init__(self, id, shares, cost):
        
#         self.id = id
#         self.shares = shares
#         self.cost = cost 

def demand(x_own, x_rest, b):
    return 100 - b * (x_own - np.sum(x_rest))


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

