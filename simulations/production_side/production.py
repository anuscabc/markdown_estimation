
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# All produciton functions check on their own before setting in the integrated market model


def log_production(beta, l, k, omega, eta):
    beta_0 = beta[0]
    beta_l = beta[1]
    beta_k = beta[2]
    y = beta_0 + beta_l*l + beta_k*k + omega + eta 
    return y


def log_labor_optimal_meval(beta, price, wage, omega, eta, k): 
    beta_0 = beta[0]
    beta_l = beta[1]
    beta_k = beta[2]
    l = (1/(beta_l-1))*(np.log((wage)/(price*beta_l))- beta_0 - omega - eta - beta_k*k)
    return l 


def investment(delta, omega, k):
    # There has to be some structure to the investment decision
    gamma = 0.1 
    # The gamma is set for all the time periods going forward 
    inv = (delta + gamma*omega)*k
    return inv


def capital_formation(delta, capital, investment):
    capital_next= (1-delta)*capital + investment
    return capital_next

def formation_capital_investment(J, T, alpha, mean_omega, sigma_omega, mean_k, sigma_k, delta):
    # Initialization fo the martix
    omega = np.zeros((J, T))
    K = np.zeros((J, T))
    inv = np.zeros((J, T))

    # Getting the first period initialization
    # There needs to be some sort of initialization for omega and capital 
    omega[:,0] = np.random.normal(mean_omega, sigma_omega, J)
    K[:, 0] = np.random.normal(mean_k, sigma_k, J)
    inv[:, 0] = investment(delta, omega[:, 0], K[:, 0])

    for t in range(1, T):
        omega[:, t] = alpha * omega[:, t-1] + np.random.normal(0, 0.1, size=J)
        K[:, t] = capital_formation(delta, K[:,t-1], inv[:,t-1])
        inv[:, t] = investment(delta, omega[:, t], K[:, t])

    return omega, K, inv



def formation_labor_production_log(J, T, K, beta, price, wage, omega, eta_labor, eta_production):
    labor = np.zeros((J, T))
    output = np.zeros((J, T))

    labor[:, 0] = log_labor_optimal_meval(beta, price[:,0], wage[:,0], omega[:, 0], eta_labor[:, 0], K[:,0])
    print(labor)
    output[:, 0] = log_production(beta, labor[:, 0], K[:, 0], omega[:, 0], eta_production[:, 0])
    print(output)

    for t in range(1, T): 
        print(t)
        labor[:, t] = log_labor_optimal_meval(beta, price[:,t], wage[:,t], omega[:, t], eta_labor[:, t], K[:,t])
        output[:, t] = log_production(beta, labor[:, t], K[:, t], omega[:, t], eta_production[:, t])

    return labor, output
     
     