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

# This is a file with all the functions that are going to be used for the estimation and called in the main


# Getting the fucntions to allow computation with teh delta parameters 
# Generalized market share function: 
def market_share_general_nohetergo(theta, df_MC):
    utility = theta[0]*df_MC["x_1"] + theta[1]*df_MC["x_2"] + theta[2]*df_MC["x_3"] + (-(np.exp(theta[3] + theta[4]*df_MC["v_p"])))*df_MC["p"] + df_MC['e']
    df_MC['u'] = utility.tolist() 
    df_max_u_index = df_MC.groupby(['t', 'i'])['u'].transform(max) == df_MC['u']
    q_i  = []
    for i in df_max_u_index:
        if i == True:
            q_i.append(1)
        else:
            q_i.append(0)
    df_MC['q'] = q_i
    copy_final_data = df_MC.copy(deep = True)
    sum_quantity_market = df_MC.groupby(['t'], as_index=False)['q'].agg('sum')
    sum_quantity_market_good = copy_final_data.groupby(['t', 'j'], as_index=False)['q'].agg('sum')
    
    # Need to put them all together in the final dataset 
    df_final1 = pd.merge(df_MC, sum_quantity_market, on=['t'], how='inner')
    df_final2 = pd.merge(df_final1, sum_quantity_market_good, on =['t', 'j'], how='inner' )
   
   # Getting the market shares for each othe goods in each market
    shares = df_final2['q']/df_final2['q_y']
    df_final2['shares'] = shares.tolist() 
    # Replace 0 shares in case very small value (see thesis notes)
    df_final2['shares'] = df_final2['shares'].replace(0, 1e-10)
    return shares, df_final2


def MC_dataset(L, T, K, M, X):
    V_MC = demand_data_simulation.consumer_heterg_data(L, T, K)
    df_MC = demand_data_simulation.merge_datasets(V_MC, M, X)
    df_MC = demand_data_simulation.get_error(df_MC)

    return df_MC
