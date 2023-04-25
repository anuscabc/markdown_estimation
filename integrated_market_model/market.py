import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


class IntegratedMarketModel:
    # definition of the method also with stype checking 
    def __init__(
            self, 
            n_firms:int, 
            n_consumers:int,
            n_chars:int,
            T:int, 
            beta=np.array([2, -0.5, -0.3]),
            mu:float=0.5, 
            omega:float=1.,
            alpha:float=0.7,
            delta:float=0.05,
            gamma:float=0.1, 
            mean_productivity:float=1.,
            std_productivity:float=.1,
            mean_capital:float=1.,
            std_capital:float=0.5,
            xi_0:float=1.,
            xi_L:float=0.3,
            xi_K:float=0.7,
            wage:float=0.5,
            seed:int=100
        ):

        np.random.seed(seed)

        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_chars = n_chars
        self.T = T

        # Parameters for demand-side simulation
        self.beta = beta
        self.mu = mu
        self.omega = omega

        # Parameters for supply-side simulation
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.mean_productivity = mean_productivity
        self.std_productivity = std_productivity
        self.mean_capital = mean_capital
        self.std_capital = std_capital
        self.xi_0 = xi_0
        self.xi_L = xi_L
        self.xi_K = xi_K
        self.wage = wage

        self.prices_0 = np.ones(self.n_firms)

        # Data structures for simulation data
        self.prices = np.zeros((self.n_firms, self.T))
        self.costs = np.zeros((self.n_firms, self.T))
        self.market_shares = np.zeros((self.n_firms, self.T))
        self.profits = np.zeros((self.n_firms, self.T))
        self.markups = np.zeros((self.n_firms, self.T))

        # Randomly generate stochastic elements
        self.produc_chars = self.gen_product_chars()
        productivity_shocks, capital, investments = self.gen_productivity_capital_investments()
        self.productivity_shocks = productivity_shocks
        self.capital = capital
        self.investments = investments
        self.labor = np.zeros((self.n_firms, T))

    def demand_side_optimisation(self):
        """Making a for loop where the cost changes each period"""

        for t in range (self.T): 

            # Generate random shocks
            v_p = np.random.normal(0, 1, size=self.n_consumers)
            e = 0.

            res1 = scipy.optimize.root(
                self.root_objective, 
                self.prices_0, 
                args=(v_p, e, t),
                method='broyden2'
            )
            self.prices[:,t] = res1.x

            market_shares, _ = self.compute_share(v_p, self.prices[:,t], e)
            self.market_shares[:,t] = market_shares
            self.costs[:,t] = self.compute_marginal_cost(market_shares, t)

            # Compute profits and markups for optimal prices
            self.compute_profit(t)
            self.compute_markup(t)

        self.compute_labor()

    def root_objective(self, price, v_p, e, t):
        """_summary_

        Args:
            price (_type_): _description_
            v_p (_type_): _description_
            e (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        market_shares, all_probs = self.compute_share(v_p, price, e)
        cost = self.compute_marginal_cost(market_shares, t)
        Jacobian = self.construct_Jacobian(all_probs, v_p) 
        profit_FOC = np.matmul(np.transpose(Jacobian), (price - cost)) + market_shares
        return profit_FOC

    def compute_marginal_cost(self, market_shares, t):
        """_summary_

        Args:
            market_shares (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        MC = (self.wage*(1/self.xi_L) *(self.n_consumers * market_shares/(np.exp(self.xi_0 + self.productivity_shocks[:,t])*
              self.capital[:,t]**self.xi_K))**((1-self.xi_L)/self.xi_L) *
              (1/np.exp(self.xi_0 + self.productivity_shocks[:,t])*
              self.capital[:,t]**self.xi_K))
        return MC

    def compute_profit(self, t):
        self.profits[:,t] = self.market_shares[:,t]*(self.prices[:,t] - self.costs[:,t])

    def compute_markup(self, t): 
        self.markups[:,t] = (self.prices[:,t] - self.costs[:,t])/self.costs[:,t]

    def compute_share(self, v_p, price, e):
        """_summary_

        Args:
            v_p (_type_): _description_
            price (_type_): _description_
            e (_type_): _description_

        Returns:
            _type_: _description_
        """
        X_for_utility = np.repeat(self.produc_chars, self.n_consumers, axis=0)
        price_r = np.reshape(price, (1, self.n_firms))
        alpha_i = np.reshape(-(np.exp(self.mu + self.omega*v_p)), (self.n_consumers, 1))
        random_coeff = np.ravel((alpha_i*price_r).T)

        u = X_for_utility@self.beta + random_coeff + e
        u_r = np.reshape(u, (self.n_firms, self.n_consumers))
        sum_u = np.sum(np.exp(u_r))

        all_probs = np.exp(u_r)/(1 + sum_u)
        market_shares = np.sum(all_probs, axis=1)

        return market_shares, all_probs

    def construct_Jacobian(self, all_probs, v_p):
        """_summary_

        Args:
            v_p (_type_): _description_
            all_probs (_type_): _description_

        Returns:
            _type_: _description_
        """
        J = np.zeros((self.n_firms, self.n_firms))
        alphas = -np.exp(self.mu + self.omega * v_p)
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                p1 = all_probs[i, :]
                if i == j:
                    J[i, j] = np.sum(alphas * p1 - alphas * p1 ** 2)
                else: 
                    p2 = all_probs[j, :]
                    J[i, j] = np.sum(alphas * p1 * p2)
        return J

    def gen_product_chars(self):
        """Generates product characteristics"""
        X1 = np.random.uniform(5, 6, size=self.n_firms)
        X2 = np.random.uniform(5, 6, size=self.n_firms)
        all_X = np.column_stack((X1, X2))
        X0 = np.ones(self.n_firms)
        return np.column_stack((X0, all_X))
    
    def compute_investment(self, productivity, capital):
        return (self.delta + self.gamma*productivity)*capital
    
    def compute_labor(self):
        # print(self.capital)
        # print(self.productivity_shocks)
        # print(self.wage)
        self.labor = (self.wage/(self.xi_L*self.capital**self.xi_K*
                      np.exp(self.xi_0 + self.productivity_shocks)*
                      self.prices))**(1/(self.xi_L-1))
        print(self.labor)
    
    def capital_formation(self, capital, investment):
        return (1-self.delta)*capital + investment

    def gen_productivity_capital_investments(self):
        """_summary_
        """

        # Initialization fo the martix
        productivity_shocks = np.zeros((self.n_firms, self.T))
        capital = np.zeros((self.n_firms, self.T))
        investments = np.zeros((self.n_firms, self.T))

        # Getting the first period initialization
        # There needs to be some sort of initialization for omega and capital 
        productivity_shocks[:,0] = np.random.normal(self.mean_productivity, self.std_productivity, self.n_firms)
        capital[:, 0] = np.random.normal(self.mean_capital, self.std_capital, self.n_firms)
        investments[:, 0] = self.compute_investment(productivity_shocks[:, 0], capital[:, 0])

        for t in range(1, self.T):
            productivity_shocks[:, t] = (self.alpha * productivity_shocks[:, t-1] 
                                         + np.random.normal(0, 0.1, size=self.n_firms))
            capital[:, t] = self.capital_formation(capital[:,t-1], investments[:,t-1])
            investments[:, t] = self.compute_investment(productivity_shocks[:, t], capital[:, t])

        return productivity_shocks, capital, investments
    
    def save_simulation_data(self):

        # This can be solved for later 
        self.products = np.tile(np.array(range(1, self.n_firms+1)), self.T-1)
        self.time = np.repeat(np.array(range(1, self.T)), J)

    def __str__(self) -> str:
        return f"Market with {self.n_firms} firms and {self.n_consumers} consumers."

