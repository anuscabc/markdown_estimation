import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp


# This folder is meant to get the dataset fully ready for the estimation 
# using pyBLP and the random coefficients model 

# There has to be a better way to write this function, figure it out later
def market_shares_outside_good(df, T, N):
    array1 = df[['j', 'shares']].to_numpy()
    index_outside_good_shares = np.where(array1[:,0] == 0)
    axis = 0
    outside_good_shares = np.take(array1, index_outside_good_shares, axis)
    outside_good_shares_reshaped = outside_good_shares.reshape(-1,2)
    t = np.array(range(1, T+1))
    markets_t = np.repeat(t, N)
    markets_t = np.reshape(markets_t, (N*T, 1))
    to_df = np.delete(np.column_stack((markets_t, outside_good_shares_reshaped)),1, 1)

    # Make this dataframe with the share of the outside goods that needs to be matched
    consumer_i = np.reshape(np.array(range(1, N+1)), (N, 1))
    repeat_consumer = np.tile(consumer_i, (T, 1))
    to_df_2 = np.column_stack((to_df, repeat_consumer))
    index = range(1, N*T+ 1, 1)
    columns = ['t', 'share_outside', 'i']
    df_outside_good = pd.DataFrame(to_df_2, index, columns)

    # Need to get rid of all the market shares j = 0 in the dataframe
    # Merging the two datasets 
    df_clean = pd.merge(df, df_outside_good, on =['t', 'i'], how='inner' )

    return df_clean

def clear_outside_good(df): 
    df.drop(df[df['j'] == 0].index, inplace = True)
    return df

def get_logarithm_share(df): 
    df['l_share_good'] = np.log(df['shares'])
    df['l_share_good_out'] = np.log(df['share_outside'])
    df['y'] = df['l_share_good'] - df['l_share_good_out']
    return df

def drop_consumer_shared(df): 
    df =  df.drop_duplicates(['shares'])
    return df

def get_rid_not_needed(df):
    df = df.drop(labels= ['i', 'v_x_1', 'v_x_2', 'v_x_3', 'v_p'], axis = 1)
    return df