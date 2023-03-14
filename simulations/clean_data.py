import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp


# This folder is meant to get the dataset fully ready for the estimation 
# using pyBLP and the random coefficients model 

def drop_consumer_shared(df): 
    df =  df.drop_duplicates(['shares'])
    return df

def get_rid_not_needed(df):
    df = df.drop(labels= ['i', 'v_x_1', 'v_x_2', 'v_x_3', 'v_p' ,'q_x', 'u_exp_x', 'u_exp_y'], axis = 1)
    return df