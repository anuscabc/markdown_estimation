import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import openpyxl

# Some global variables 
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



df = pd.read_csv('data/data_to_run_estimaton.csv')

df2 =  df.drop_duplicates(['shares'])
df3 = df2.drop(labels= ['i', 'v_x_1', 'v_x_2', 'v_x_3', 'v_p', 'u_exp_x', 'u_exp_y'], axis = 1)

# Getting a vector of all the shares of the outside good
array1 = df3[['j', 'shares']].to_numpy()


# if (array1[:,0] == 0).all(): 
#     outside_good_shares = (array1[:,1])

# This gets all the indeces for which we are looking at the outside goods 
index_outside_good_shares = np.where(array1[:,0] == 0)
axis = 0
outside_good_shares = np.take(array1, index_outside_good_shares, axis)
outside_good_shares_reshaped = outside_good_shares.reshape(-1,2)


# Making them coincide with each of the markets they were in 

markets_t = np.array(range(1, T+1, 1))
to_df = np.delete(np.column_stack((markets_t, outside_good_shares_reshaped)),1, 1)
# Make this dataframe with the share of the outside goods that needs to be matched
index = range(1,T+ 1, 1)
columns = ['t', 'share_outside']
df_outside_good = pd.DataFrame(to_df, index, columns)

# Need to get rid of all the market shares j = 0 in the dataframe

df3.drop(df3[df3['j'] == 0].index, inplace = True)
print(df3)

# Make also a new column in the market with the total quanroty in the market with
# without the outside good observation 


# Merging the two datasets 
df_clean = df3.merge(df_outside_good, on='t', how = 'left')

#Maybe also need some 

df_clean.to_csv()
df_clean.to_excel('data/data_clean.xlsx')

