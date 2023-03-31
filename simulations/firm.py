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
# This should work for normalized consumer mass to 1 
p = np.array([1., 1., 0])
c = np.array([0.4, 0.5, 0])
X = np.array([0.3, 0.4, 0])
beta = [1, 2]
alpha = -0.5
# initiate some number of consumers 
N = 1
e = np.random.gumbel(0, 1, size = len(p*N))


def sum_utility(beta, X, price, alpha, e):
    # here think a bit about how you want to introduce the outside good 
    sum_u_exp = sum((np.exp(beta[0] + alpha*price + beta[1]*X+e)[:-1]))
    return sum_u_exp

def probability(beta, X, price, alpha, sum_u_ex, e): 
    q = (np.exp(beta[0] + alpha*price + beta[1]*X+ e))/(1 + sum_u_ex)
    return q

def profit(beta, X, price, alpha, sum_u_ex, e, c):
    q = probability(beta, X, price, alpha, sum_u_ex, e)
    Pi = - 1*q*(price - c)
    return Pi

def inverse(beta, X, price, alpha, sum_u_ex, e, c):


# Need to rewrite and get the inverse for the fixed point inverse function for the iteration 

# in this case this has to be: 
# somethink along the lines of: 
# need to make check that the determinant of the matrix is non zero but there you already work 
# with a vector cause each frm produced only one good 
def fixedPointIteration(price, error, num_ter):
    print('\n\n*** FIXED POINT ITERATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        #THERE WE NEED TO HAVE THE INPUT TO CALL ON THE INVERSE FUNCTION 
        price1 = c
        print('Iteration-%d, price = %0.6f and profit(x1) = %0.6f' % (step, x1, f(x1)))
        price = price1

        step = step + 1
        
        if step > num_ter:
            flag=0
            break
        
        condition = abs(profit(_funciton of all in_)) > error

    if flag==1:
        print('\nRequired root is: %0.8f' % price)
    else:
        print('\nNot Convergent.')




sum = sum_utility(beta, X, p, alpha, e)
prob = probability(beta, X, p, alpha, sum, e)
prof = profit(beta, X, p, alpha, sum, e, c)


print(prob)
print(prof)

