import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import firm

np.random.seed(8)
# The number of firms in the market
N = 10

# The number of product characteristics
J = 2

# put in the true param because the firm actually knows them
beta = np.array([2., -0.611, -0.528])
alpha = -2.71


p = np.ones(N)
c = np.random.lognormal(0, 1, size = N)
Xj = np.random.lognormal(0, 1, size = (N, J)) 
X0 = np.ones(N)
X = np.column_stack((X0, Xj))


res1 = scipy.optimize.root(firm.root_objective, p, args=(N, X, beta, alpha, c), method = 'broyden2')
optimal_price = res1.x
print(optimal_price)
