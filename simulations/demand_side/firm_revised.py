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
    u = X_for_utility@coef_car+ random_coeff + e
    u_r = np.reshape(u, (J_prod, N_cons))
    sum_u = np.sum(np.exp(u_r))
    prob = np.exp(u_r)/(1 +sum_u)
    shares = np.sum(prob, axis=1)
    return(shares)


# Need to write some sort of function here to convince moraga that the continous function 
# and the discrete function ultimately lead to the same sort of 
# def discrete_share():
#     return 

def derivative(J_prod, mu, omega, v_p, shares):
    J = np.zeros((J_prod, J_prod))
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if i == j:
                # this is the own product elasticity
                J[i, j] = np.sum((-(np.exp(mu + omega*v_p)))* shares[i] - (-(np.exp(mu + omega*v_p)))*shares[i]**2)
            elif i!= j: 
                # this is the cross product elasticity
                J[i, j] = np.sum(-(-(np.exp(mu + omega*v_p)))*shares[i]*shares[j])
    return J

def root_objective(price, N_cons, J_prod, X_char, v_p, coef_car, mu, omega, e, c):
    X_for_utility = np.repeat(X_char, N_cons, axis=0)
    price_r = np.reshape(price, (1, J_prod))
    alpha_i = np.reshape(-(np.exp(mu + omega*v_p)), (N_cons, 1))
    random_coeff = np.ravel((alpha_i*price_r).T)
    u = X_for_utility@coef_car+ random_coeff + e
    u_r = np.reshape(u, (J_prod, N_cons))
    sum_u = np.sum(np.exp(u_r))
    prob = np.exp(u_r)/(1 +sum_u)
    shares = np.sum(prob, axis=1)
    Jacobian = derivative(J_prod, mu, omega, v_p, shares) 
    optim = np.matmul(np.transpose(Jacobian), (price - c)) + shares
    return optim