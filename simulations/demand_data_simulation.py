import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
sd_x = 0.5
sd_c = 0.05
sd_p = 0.05

# # # # # 
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



# Getting the price marginal cost and the exognous demand shock in a dataframe 
x_i = np.zeros(shape = (J*T, 1))
c = np.random.lognormal(mean=0.0, sigma= sd_c, size=(J*T, 1))
p =  c+ np.random.lognormal(price_xi*x_i, sd_p, size=(J*T, 1))
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
columns = ["j", "t", "x_i", "c", "p"]


df_price_cost = pd.DataFrame(M, index , columns)
grouped = df_price_cost.groupby("t")
df_price_cost_sample_group = grouped.sample(frac = 1-prop_jt)
df_price_cost = df_price_cost_sample_group.reset_index()
print(df_price_cost)


# Putting in the outside option for each of the markets 
new_row = {'j':0, 'x_i':0, 'c':0, 'p':0}
df2 = 



