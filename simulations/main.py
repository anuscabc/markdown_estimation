import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats 
import matplotlib.pyplot as plt
# importing the data simulation functions 
import demand_data_simulation
import clean_data
import statsmodels.api as sm
import statsmodels.formula.api as smf



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
L = 300 

# Setting the parameter of interest 
beta = np.random.normal(0, 1, K)

# Here put whatever constant you want 
beta[0]= 2

sigma = np.absolute(np.random.normal(0, 1, K))


# Generate the parameters for all that has to be differntiated among the
# consumers 
mu = 0.5 
omega = 1 


# Setting some anxilliary parameters 
# set auxiliary parameters
price_xi = 1 # This is the gamma in the MacFadden paper()
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
M = demand_data_simulation.get_price_cost_data(J, T, price_xi, sd_c, sd_p)
M = demand_data_simulation.getting_rid_random_products(M)
M = demand_data_simulation.outside_good_consumer_choice_data(M, T)

# 3. Get the consumer schocks 
V = demand_data_simulation.consumer_heterg_data(N, T, K)

# 4. Get the full dataset 
# DO NOT CHANGE ORDER ARGUMENTS OFTHERWISE LEFT MERGE NOT WORK! 
df = demand_data_simulation.merge_datasets(V, M, X)

# 5. Put the continous quantity in the dataframe 
df = demand_data_simulation.get_continous_quantity(df, mu , omega, sigma, beta)

# 6. Getting the market share 
df = demand_data_simulation.get_market_shares(df)


# 7. Getting the outside good_market share as an additional variable  
# Making the dataset all clean but also ready to run the monte-carlo simulations 

df = demand_data_simulation.market_shares_outside_good(df, T, N)

# 8. Getting rid of the outside good within the dataset 
df = demand_data_simulation.clear_outside_good(df)

# 9. Get logatirhm variable 
df = demand_data_simulation.get_logarithm_share(df)
# Trying estimation with the PyBLP package for 
# random coefficient logit 

df = clean_data.drop_consumer_shared(df)
df = clean_data.get_rid_not_needed(df)
df.to_csv("data/estimation1.csv")


# ## Cool graph actual product mark-up denerated by the model at hand 

# mark_up = df['p'] - df['c']


# # A mark-up histogram
# hist = plt.hist(mark_up, bins=100)  # arguments are passed to np.histogram
# plt.title("Histogram markup")
# plt.show()


# # Get the error 
# hist2 = plt.hist(df['shares'], bins=100) 
# plt.title("Histogram aggregate market shares")
# plt.show()

# Getting started with the optimization algorthm and monte-carlo simluation for the 
# market shares 
print(beta)
print(sigma)








