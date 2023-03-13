import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
# importing the data simulation functions 
import demand_data_simulation


# Seeting the seed for the simulation 
np.random.seed(1)


# Getting the number of products 
J = 10 

# Dimmention product characteristics 
K = 3 # 2 with random coeff and the other constant also random coeff

# Number of markets 
T = 100 

# Number of consumsers per market 
N = 500 

# This need to think more about, now it is continous quantity, but is that 
# the way to go? Need to ask this on Friday
L = 500 

# Setting the parameter of interest 
beta = np.random.normal(0, 1, K)

# Here put whatever constant you want 
beta[0]= 4

sigma = np.absolute(np.random.normal(0, 1, K))


# Generate the parameters for all that has to be differntiated among the
# consumers 
mu = 0.5 
omega = 1 


# Setting some anxilliary parameters 
# set auxiliary parameters
price_xi = 1 # This is the gamma in the MacFadden paper()
prop_jt = 0.6
sd_xi = 0.1
sd_x = 0.1
sd_c = 0.5
sd_p = 0.01


# PLEASE NOTE THAT THE COLUMN NAMES HAVE TO BE CHANGED IN THE 
# PRODUCT CHARACT DATASET DEPENDING ON THE K THAT YOU ARE CHOOSING!!!! 
# ALL the columns in all of the functions 


# 1. Get the dataset for the products 
# This X is actually a dataframe 
X = demand_data_simulation.get_product_market_data(J, K, sd_x)

# 2. Get the costs and the prices 