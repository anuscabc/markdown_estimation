import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
# import scipy.optimize
import matplotlib.pyplot as plt
# importing the data simulation functions 
import demand_data_simulation
import clean_data
import demand_data_estimation
import statsmodels.api as sm
import statsmodels.formula.api as smf


pd.set_option('float_format', '{:.2f}'.format)


# Seeting the seed for the simulation 
np.random.seed(1)

# Getting the number of products 
J = 10

# Dimmention product characteristics 
K = 3 # 2 with random coeff and the other constant also random coeff

# Number of markets 
T = 100

# Number of consumers per market 
N = 500

# This need to think more about, now it is continous quantity, but is that 
# the way to go? Need to ask this on Friday
L = 500

# Setting the parameter of interest 
beta = np.random.normal(0, 1, K)

# Here put whatever constant you want 
beta[0]= 2
# This is for the case where more than one
# parameter with random coefficients 
# sigma = np.absolute(np.random.normal(0, 1, K))

# Generate the parameters for all that has to be differntiated among the
# consumers 
mu = 0.5 
omega = 1 


# Setting some anxilliary parameters 
# set auxiliary parameters
# T=
price_xi = 1 # This is the gamma in the MacFadden paper()
sd_xi = 0.1


sd_x = 0.1
sd_c = 0.5
sd_p = 0.01


# PLEASE NOTE THAT THE COLUMN NAMES HAVE TO BE CHANGED IN THE 
# PRODUCT CHARACT DATASET DEPENDING ON THE K THAT YOU ARE CHOOSING!!!! 
# ALL the columns in all of the functions 

# 1. Generate initial dataset from which we can do all the estimation
output = demand_data_simulation.generate_demand_data(J, K, T, N, price_xi, sd_x, sd_c, sd_p)
df = output[0]
X = output[1]
M = output[2]
V = output[3]



# 2. Get the error term calculating the utility
df = demand_data_simulation.get_error(df)

# 3. Put the discrete quantity in the dataframe 
df = demand_data_simulation.get_discrete_quantity(df, mu , omega, beta)

# 4. Getting the market share 
df = demand_data_simulation.get_market_shares(df)


# Parameter estimation assuming no endogenous prices
theta_true = np.append(beta, [mu, omega])
theta_0 = [1.2, -0.5, -0.6, 0.4, 1.]

np.random.seed(2)

df_MC = demand_data_estimation.MC_dataset(L, T, K, M, X)

def f(theta, df_MC, df):
    shares_true = df['shares'].to_numpy()
    shares_est, _ = demand_data_estimation.market_share_general_nohetergo(theta, df_MC)
    return np.linalg.norm(shares_true - shares_est.to_numpy())


res = minimize(f, theta_0, args=(df_MC, df), method = 'Nelder-Mead')
print(res.x)
print(theta_true)



 




