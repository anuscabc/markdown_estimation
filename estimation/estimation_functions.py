import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import estimation_functions



def simulate_market_shares_per_period(theta, df, v_p, n_consumers, n_firms, t): 
    """_summary_

    Args:
        theta (_type_): _description_
        df (_type_): _description_
        v_p (_type_): _description_
        n_consumers (_type_): _description_
        n_firms (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    beta = theta[:3]
    mu = theta[3]
    omega = theta[4]
    X0 = np.ones(n_firms)
    X1 = df['characteristic1'].head(n_firms)
    X2 = df['characteristic2'].head(n_firms)
    X_car = np.column_stack((X1, X2))
    X = np.column_stack((X0, X_car))


    price_r = np.reshape(np.array(df.prices[t*n_firms:(t+1)*n_firms]), (1, n_firms))
    # price_r = np.reshape(np.array(df.predict_prices[t*n_firms:(t+1)*n_firms]), (1, n_firms))
    alpha_0 = -np.exp(mu + omega**2/2)
    mean_indirect_utility = X@beta + alpha_0*np.array(df.prices[t*n_firms:(t+1)*n_firms])
    # mean_indirect_utility = X@beta + alpha_0*np.array(df.predict_prices[t*n_firms:(t+1)*n_firms])
    mean_indirect_utlity_for_utility = np.repeat(mean_indirect_utility, n_consumers, axis=0)

    alpha_i = np.reshape((-(np.exp(mu +omega*v_p))+np.exp(mu +omega**2/2)), (n_consumers, 1))
    random_coeff = np.ravel((alpha_i*price_r).T)

    u = mean_indirect_utlity_for_utility + random_coeff 

    # X_for_utility = np.repeat(X, n_consumers, axis=0)
    # price_r = np.reshape(df.prices[t*n_firms:(t+1)*n_firms], (1, n_firms))
    # alpha_i = np.reshape(-(np.exp(mu + omega*v_p)), (n_consumers, 1))
    # random_coeff = np.ravel((alpha_i*price_r).T)

    # u = X_for_utility@beta + random_coeff

    u_r = np.reshape(u, (n_firms, n_consumers))
    sum_u = np.sum(np.exp(u_r))

    all_probs = np.exp(u_r)/(1 + sum_u)
    market_shares = np.sum(all_probs, axis=1)
    
    return market_shares


def get_indirect_utility(theta, df, n_firms, t):
    """_summary_

    Args:
        theta (_type_): _description_
        df (_type_): _description_
        n_consumers (_type_): _description_
        n_firms (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    beta = theta[:3]
    mu = theta[3]
    omega = theta[4]
    X0 = np.ones(n_firms)
    X1 = df.characteristic1.head(n_firms)
    X2 = df.characteristic2.head(n_firms)
    X_car = np.column_stack((X1, X2))
    X = np.column_stack((X0, X_car))

    # X_for_utility = np.repeat(X, n_consumers, axis=0)
    price_r = np.array(df.prices[t*n_firms:(t+1)*n_firms])
    alpha = -(np.exp(mu + omega**2/2))
    # random_coeff_mean = np.ravel((alpha*price_r).T)

    indirect_utility = X@beta + alpha*price_r
    return indirect_utility
