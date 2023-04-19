import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def profit(price, share, cost):
    Pi = share*(price - cost)
    return Pi

def markup(price, cost): 
    markup = (price - cost)/cost
    return markup

# Making a killer function for calculating market shares 
def share(N_cons, J_prod, X_char, v_p, price, coef_car, mu, omega, e): 
    X_for_utility = np.repeat(X_char, N_cons, axis=0)
    price_r = np.reshape(price, (1, J_prod))
    alpha_i = np.reshape(-(np.exp(mu + omega*v_p)), (N_cons, 1))
    random_coeff = np.ravel((alpha_i*price_r).T)
    u = X_for_utility@coef_car + random_coeff + e
    u_r = np.reshape(u, (J_prod, N_cons))
    sum_u = np.sum(np.exp(u_r))
    all_probs = np.exp(u_r)/(1 +sum_u)
    shares_per_firm = np.sum(all_probs, axis=1)
    return shares_per_firm, all_probs


# Need to write some sort of function here to convince moraga that the continous function 
# and the discrete function ultimately lead to the same sort of 
# def discrete_share():
#     return 


def construct_Jacobian(J_prod, mu, omega, v_p, all_probs):

    J = np.zeros((J_prod, J_prod))
    alphas = -np.exp(mu + omega * v_p)
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            p1 = all_probs[i, :]
            if i == j:
                J[i, j] = np.sum(alphas * p1 - alphas * p1 ** 2)
            else: 
                p2 = all_probs[j, :]
                J[i, j] = np.sum(alphas * p1 * p2)
    return J


def root_objective(price, N_cons, J_prod, X_char, v_p, coef_car, mu, omega, e, c):
    shares_per_firm, all_probs = share(N_cons, J_prod, X_char, v_p, price, coef_car, mu, omega, e)
    Jacobian = construct_Jacobian(J_prod, mu, omega, v_p, all_probs) 
    profit_FOC = np.matmul(np.transpose(Jacobian), (price - c)) + shares_per_firm
    return profit_FOC