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




pd.set_option('float_format', '{:.2f}'.format)


# Seeting the seed for the simulation 
np.random.seed(1)

# Getting the number of products 
J = 3

# Dimmention product characteristics 
K = 3 # 2 with random coeff and the other constant also random coeff

# Number of markets 
T = 10

# Number of consumers per market 
N = 5

# This need to think more about, now it is continous quantity, but is that 
# the way to go? Need to ask this on Friday
L = 5

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
df = demand_data_simulation.get_error(df)


# 5. Put the continous quantity in the dataframe 
df_1 = demand_data_simulation.get_continous_quantity(df, mu , omega, beta)
df = demand_data_simulation.get_discrete_quantity(df, mu , omega, beta)

# 6. Getting the market share 
df = demand_data_simulation.get_market_shares(df)
df_1 = demand_data_simulation.get_market_shares(df_1)


theta_true = np.append(beta, [mu, omega])

# This is now calculated at the true parameter 
df_1 = demand_data_simulation.delta(theta_true, df)

# Rethink this a bit, might be easier to just make some instruments and use the pyblp package!! 





