import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp


# Seeting the seed for the simulation 
np.random.seed(1)


# Getting the number of products 
J = 10 

# Dimmention product characteristics 
K = 3 

# Number of markets 
T = 100 

# Number of conumsers per market 
N = 500 

# number of monte Carlo 
L = 500 

# Setting the parameter of interest 
beta = np.random.normal(0, 1, K)
beta[0]= 4

sigma = np.absolute(np.random.normal(0, 1, K))

# Generate the parameters for all that has to be differntiated among the
# consumers 

mu = 0.5 
omega = 1 


# Setting some anxilliary parameters 
# set auxiliary parameters
price_xi = 1
prop_jt = 0.6
sd_xi = 0.1
sd_x = 0.1
sd_c = 0.5
sd_p = 0.01

# # # # # 
#

def get_product_market_data(J, K, sd_x):
# Getting the dataframe with the product market characteristics (X)
    X = sd_x *(np.random.normal(0, 1, size = (J, (K - 1))))
    X_1 = np.ones(shape= (J, 1))
    outside_option = np.zeros(shape = (1, K+1))
    j = np.reshape(np.array(range(1, J+1)), (J, 1))
    X = np.column_stack((X_1, X))
    X = np.column_stack((j, X))
    X = np.concatenate((outside_option, X))
    index = range(1, J+2)
    # The column names have to be changed depending on the 
    # number of product characteristics that have to be included 
    # in the estimation if it doesnt work that is why 
    columns = ["j", "x_1", "x_2", "x_3"]
    df_product_car = pd.DataFrame(data = X, index = index, columns= columns )
    return df_product_car
    

def get_price_cost_data(J, T, price_xi, sd_c, sd_p):
    # Getting the price marginal cost and the exognous demand shock in a dataframe 
    x_i = np.random.normal(0, 1, size = (J*T, 1))
    c = np.random.lognormal(mean=0.0, sigma= sd_c, size=(J*T, 1))
    p =  c+ np.random.lognormal(price_xi*x_i, sd_p, size=(J*T, 1))
    j = np.reshape(np.array(range(1, J+1)), (J, 1))
    repeat_j = np.tile(j, (T, 1))
    t = np.array(range(1, T+1))
    repeat_t = np.repeat(t, J)
    repeat_t = np.reshape(repeat_t, (J*T, 1))
    M = np.column_stack((repeat_j, repeat_t))
    M = np.column_stack((M, x_i))
    M = np.column_stack((M, c))
    M = np.column_stack((M, p))
    # Getting the dataframe
    index = range(1, J*T+ 1)
    # The column names have to be changed depending on the 
    # number of product characteristics that have to be included 
    # in the estimation if it doesnt work that is why 
    columns = ["j", "t", "xi", "c", "p"]
    df_price_cost = pd.DataFrame(M, index , columns)
    grouped = df_price_cost.groupby("t")
    # # The fraction kept in the dataset has to be different for each of the given market 
    df_price_cost_sample_group = grouped.sample(frac = 1)
    df_price_cost = df_price_cost_sample_group.reset_index(drop = True)
    # Getting the outside option for the price_cost_dataframe 
    for i in range(1, T+1, 1): 
        l = [0, i, 0, 0, 0]
        df_price_cost.loc[len(df_price_cost.index)] = l
    sorted = df_price_cost.sort_values('t')
    df_price_cost = sorted.reset_index(drop = True)
    return df_price_cost


# Generating the consumer heterogeneity dataset 

def consumer_heterg_data(N, T, K):
    consumer_i = np.reshape(np.array(range(1, N+1)), (N, 1))
    repeat_consumer = np.tile(consumer_i, (T, 1))
    t_consumer_repeat = np.repeat(t, N)
    t_consumer_repeat = np.reshape(t_consumer_repeat, (N*T, 1))
    shocks = np.random.normal(0, 1, size=(N*T, K+1))
    V = np.column_stack((repeat_consumer, t_consumer_repeat))
    V = np.column_stack((V, shocks))
    # Getting the dataframe
    index = range(1, N*T+ 1)
    columns = ['i', 't' , 'v_x_1', 'v_x_2', 'v_x_3', 'v_p']
    df_shocks_price = pd.DataFrame(V, index, columns)
    return df_shocks_price 



# Merging all the datasets together for a clean anf nice full dataset 
def merge_datasets(df_1, df_2, df_3): 
    merge1 = df_1.merge(df_2, on='t', how = 'left')
    df_total = merge1.merge(df_3, on = 'j', how ='left')
    return df_total


# Getting the error in the dataset but idk this maybe should be done 
# differently depending on how you want to include the 
def get_error(df):
    # Getting the extreme value distribution vector 
    e = np.random.gumbel(0, 1, size = len(df))
    df['e'] = e.tolist()
    return df


 


# Next step for tmr is to get the indirect utility utility, quantity and the market shares 
# Getting also the utility in the dataframe and the quantity 


alpha_i = -(np.exp( mu + omega*df_total["v_p"]))
beta_1i = beta[0] + sigma[0]*df_total["v_x_1"]
beta_2i = beta[1] + sigma[1]*df_total["v_x_2"]
beta_3i = beta[2] + sigma[2]*df_total["v_x_3"]

u_i = beta_1i*df_total["x_1"] + beta_2i*df_total["x_2"] + beta_3i*df_total["x_3"] + alpha_i*df_total["p"] + df_total["x_i"]
df_total['u'] = u_i.tolist() 

exp_u_i = np.exp(df_total['u'])
df_total['u_exp'] = exp_u_i.tolist() 


# Getting the quantity each consumer gets in each of the markets 
sum_utility_group = df_total.groupby(['t','i'], as_index=False)['u_exp'].agg('sum')
df_final = pd.merge(df_total, sum_utility_group, on=['t','i'], how='inner')


# df_total_utilities = sum_utility_group.reset_index(drop = True)

q = df_final['u_exp_x']/(1 + df_final['u_exp_y'])
df_final['q'] = q.tolist() 

# print(df_final.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False))

# Now need to get the actual shares and the difference in the log of the shares 
# log s_jt = log_share outside good in market T 
# Getting the sum of the quantities in each market for each good 

copy_final_data = df_final.copy(deep = True)


sum_quantity_market = df_final.groupby(['t'], as_index=False)['q'].agg('sum')
sum_quantity_market_good = copy_final_data.groupby(['t', 'j'], as_index=False)['q'].agg('sum')


# Need to put them all together in the final dataset 
df_final1 = pd.merge(df_final, sum_quantity_market, on=['t'], how='inner')
df_final2 = pd.merge(df_final1, sum_quantity_market_good, on =['t', 'j'], how='inner' )

# Getting the market shares for each othe goods in each market
shares = df_final2['q']/df_final2['q_y']

df_final2['shares'] = shares.tolist() 



# Getting the data out for running all the simulations 
df_final2.to_csv("data/data_to_run_estimaton.csv")

# This needs to be rewritten as a function and with a main in
# order to get everything in order 