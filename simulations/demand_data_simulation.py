import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

# This is all the functions for generating the dataset without the manipulation for the outside good market share


def get_product_market_data(J, K, sd_x):
# Getting the dataframe with the product market characteristics (X)
    X = sd_x * (np.random.normal(0, 1, size = (J, (K - 1))))
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
    # This is for when prices are not endogenous 
    x_i = np.zeros(shape = (J*T, 1))

    # This is for getting the endogenous prices 
    # x_i = np.random.normal(0, 1, size = (J*T, 1))


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
    # Getting the outside option for the price_cost_dataframe 
    return df_price_cost


def getting_rid_random_products(df_price_cost):
    grouped = df_price_cost.groupby("t")
    # # The fraction kept in the dataset has to be different for each of the given market 
    df_price_cost_sample_group = grouped.sample(frac = 1)
    df_price_cost = df_price_cost_sample_group.reset_index(drop = True)
    return df_price_cost


def outside_good_consumer_choice_data(df_price_cost, T):
    for i in range(1, T+1, 1): 
        l = [0, i, 0, 0, 0]
        df_price_cost.loc[len(df_price_cost.index)] = l
    sorted = df_price_cost.sort_values('t')
    df_price_cost = sorted.reset_index(drop = True)
    return df_price_cost


# Generating the consumer heterogeneity dataset 

def consumer_heterg_data(N, T, K):
    # This generalized for when there are all random coefficient stuff
    consumer_i = np.reshape(np.array(range(1, N+1)), (N, 1))
    repeat_consumer = np.tile(consumer_i, (T, 1))
    t = np.array(range(1, T+1))
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
    merge1 = df_1.merge(df_2, on= 't', how = 'left')
    df_total = merge1.merge(df_3, on = 'j', how ='left')
    return df_total


# Getting the error in the dataset but idk this maybe should be done 
# differently depending on how you want to include the 
def get_error(df):
    # Getting the extreme value distribution vector 
    e = np.random.gumbel(0, 1, size = len(df))
    df['e'] = e.tolist()
    return df


def generate_demand_data(J, K, T, N, price_xi, sd_x, sd_c, sd_p):

    X = get_product_market_data(J, K, sd_x)

    # 2. Get the costs and the prices 
    M = get_price_cost_data(J, T, price_xi, sd_c, sd_p)
    M = getting_rid_random_products(M)
    M = outside_good_consumer_choice_data(M, T)

    # 3. Get the consumer schocks 
    V = consumer_heterg_data(N, T, K)
    df = merge_datasets(V, M, X)
    return df, X, M, V





def get_continous_quantity(df,mu,omega, beta): 
    alpha_i = -(np.exp(mu + omega*df["v_p"]))
    # This matters wether or not you include sigma in the equation
    # This is for no other coefficients wuth random coefficents 
    beta_1i = beta[0] 
    beta_2i = beta[1]
    beta_3i = beta[2] 

    # Here we can change which coefficients have random coefficients
    # beta_1i = beta[0] + sigma[0]*df["v_x_1"]
    # beta_2i = beta[1] + sigma[1]*df["v_x_2"]
    # beta_3i = beta[2] + sigma[2]*df["v_x_3"]
    u_i = beta_1i*df['x_1'] + beta_2i*df["x_2"] + beta_3i*df["x_3"] + alpha_i*df["p"] + df["xi"]  + df['e']
    df['u'] = u_i.tolist() 
    exp_u_i = np.exp(df['u'])
    df['u_exp'] = exp_u_i.tolist() 

    # Getting the quantity each consumer gets in each of the markets 
    sum_utility_group = df.groupby(['t','i'], as_index=False)['u_exp'].agg('sum')
    df_final = pd.merge(df, sum_utility_group, on=['t','i'], how='inner')
    # q = max(u) take 1 else 0 
    q = df_final['u_exp_x']/(1 + df_final['u_exp_y'])
    df_final['q'] = q.tolist() 
    return df_final


def get_discrete_quantity(df, mu, omega, beta): 
    alpha_i = -(np.exp(mu + omega*df["v_p"]))
    df['alpha_i'] = alpha_i.tolist() 
    # This matters wether or not you include sigma in the equation 
    # This is for no other coefficients wuth random coefficents 
    beta_1i = beta[0] 
    beta_2i = beta[1]
    beta_3i = beta[2] 
    # Here we can change which coefficients have random coefficients
    # beta_1i = beta[0] + sigma[0]*df["v_x_1"]
    # beta_2i = beta[1] + sigma[1]*df["v_x_2"]
    # beta_3i = beta[2] + sigma[2]*df["v_x_3"]
    u_i = beta_1i*df["x_1"] + beta_2i*df["x_2"] + beta_3i*df["x_3"] + alpha_i*df["p"] + df["xi"] + df['e']
    df['u'] = u_i.tolist() 
    df_max_u_index = df.groupby(['t', 'i'])['u'].transform(max)== df['u']
    q_i  = []
    for i in df_max_u_index:
        if i == True:
            q_i.append(1)
        else:
            q_i.append(0)
    df['q'] = q_i
    return df


def get_market_shares(df):
    copy_final_data = df.copy(deep = True)
    sum_quantity_market = df.groupby(['t'], as_index=False)['q'].agg('sum')
    sum_quantity_market_good = copy_final_data.groupby(['t', 'j'], as_index=False)['q'].agg('sum')
    # Need to put them all together in the final dataset 
    df_final1 = pd.merge(df, sum_quantity_market, on=['t'], how='inner')
    df_final2 = pd.merge(df_final1, sum_quantity_market_good, on =['t', 'j'], how='inner' )
    # Getting the market shares for each othe goods in each market
    shares = df_final2['q']/df_final2['q_y']
    df_final2['shares'] = shares.tolist() 
    # Replace 0 shares in case very small value (see thesis notes)
    df_final2['shares'] = df_final2['shares'].replace(0, 1e-10)
    return df_final2








# Start by getting the indirect utility at the true parameters 
# 
def delta(theta, df):
    delta = theta[0]*df['x_1'] + theta[1]*df['x_2'] + theta[2]*df['x_3'] + (-(np.exp(theta[3] + theta[4]**2/2)))*df["p"]+ df['xi'] 
    df['delta_jt'] = delta.tolist()
    return df


# Ultimately integrate it into a function 
# def market_share_general_hetergo(theta, df):
# # This should give the mean indirect utility per product
# # This means that this should be J*T long 
#     return shares, df_final2

