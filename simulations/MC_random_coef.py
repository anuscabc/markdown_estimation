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


# This might not be needed idk (talk about this tmr)

#1. Drawing monte_carlo consumer heterogeneity
# This is supposed to give a slighly different dataset and it appears to be working at 
# the moment 
V_mcmc = demand_data_simulation.consumer_heterg_data(L, T, K)
print(V_mcmc)
df_mcmc = demand_data_simulation.merge_datasets(V_mcmc, M, X)
df_mcmc = demand_data_simulation.get_continous_quantity(df_mcmc, mu , omega, sigma, beta)
# 6. Getting the market share at the actual parametrs, but in real life I dont
# actually know these parameters so I need to somhow approximate them
#=> That is where that V_mcmc comes from -> they are not the actual shocks that 
# we have observed previously, but are going to be some randomly generated schocks 
# that are intentionally wrong 
df_mcmc = demand_data_simulation.get_market_shares(df_mcmc)
# 7. Getting the outside good_market share as an additional variable  
# Making the dataset all clean but also ready to run the monte-carlo simulations 
df_mcmc = demand_data_simulation.market_shares_outside_good(df_mcmc, T, L)
# 8. Getting rid of the outside good within the dataset 
df_mcmc = demand_data_simulation.clear_outside_good(df_mcmc)
# 9. Get logatirhm variable 
df_mcmc = demand_data_simulation.get_logarithm_share(df_mcmc)
# Trying estimation with the PyBLP package for 
# random coefficient logit 
df_mcmc = clean_data.drop_consumer_shared(df_mcmc)
df_mcmc = clean_data.get_rid_not_needed(df_mcmc)


# Need to write a function that first computes the simulated share 
# and then computes the mean sqare error with the share data
#


model = sm.MixedLM.from_formula('y ~ x_2 + x_3 + p', 
                                data=df_mcmc, 
                                groups=df_mcmc['j'], 
                                re_formula='~ x_2 + x_3 + p')

result = model.fit()
print(result.summary())