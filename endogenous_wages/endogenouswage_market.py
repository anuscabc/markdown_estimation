import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt



class EndogenousWageMarketModel:
    # definition of the method also with stype checking 
    """_summary_
    """
    def __init__(
            self, 
            n_firms:int, 
            n_consumers:int,
            n_chars:int,
            T:int, 
            beta1:float=2.,
            beta2:float=-0.3,
            beta3:float=-0.3,
            mu:float=0.5, 
            sigma:float=0.2,
            x1_min:float=1.,
            x1_max:float=6.,
            x2_min:float=1.,
            x2_max:float=6.,
            rho:float=0.7,
            tau:float=0.05,
            gamma:float=0.1, 
            mean_productivity:float=0,
            std_productivity:float=0.05,
            min_capital:float=15.,
            max_capital:float=15.,
            theta_0:float=1.,
            theta_L:float=0.6,
            theta_K:float=0.4,
            labor_elasticity = 0.01,
            seed:int=100
        ):

        self.seed = seed
        np.random.seed(self.seed)

        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_chars = n_chars
        self.T = T

        # Parameters for demand-side simulation
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.mu = mu
        self.sigma = sigma

        # parameters for the characterisitcs 
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max

        # Parameters for supply-side simulation
        self.tau = tau
        self.rho = rho
        self.gamma = gamma
        self.mean_productivity = mean_productivity
        self.std_productivity = std_productivity
        self.min_capital = min_capital
        self.max_capital = max_capital
        self.theta_0 = theta_0
        self.theta_L = theta_L
        self.theta_K = theta_K
        
        # This is the new variable for the labor elasticity 
        self.labor_elasticity = labor_elasticity

        # Initial price vector
        self.prices_0 = np.ones(self.n_firms)

        # Data structures for simulation data
        self.prices = np.zeros((self.n_firms, self.T))
        self.costs = np.zeros((self.n_firms, self.T))
        self.market_shares = np.zeros((self.n_firms, self.T))
        self.mean_indirect_utilities = np.zeros((self.n_firms, self.T))
        self.profits = np.zeros((self.n_firms, self.T))
        self.markups = np.zeros((self.n_firms, self.T))
        self.wages = np.zeros((self.n_firms, self.T))

        # All random consumer level shocks for prices
        self.v_p = np.random.normal(0, 1, (self.n_consumers, self.T))


        # Supply side data
        productivity_shocks, capital, investments = self.gen_productivity_capital_investments()
        self.productivity_shocks = productivity_shocks
        self.capital = capital
        self.investments = investments
        self.labor_quantity = np.zeros((self.n_firms, T))

        # Randomly generate stochastic elements
        self.produc_chars = self.gen_product_chars()


    def demand_side_optimisation(self):
        """ The prices are going to be optimally set in each given period """

        for t in range (self.T): 

            # Generate random shocks
            v_p_r = self.v_p[:,t]
            e = 0.
            res1 = scipy.optimize.root(
                self.root_objective, 
                self.prices_0, 
                args=(v_p_r, e, t),
                method='broyden2',
            )
            self.prices[:,t] = res1.x

            market_shares, probabilities, mean_indirect_util = self.compute_share(v_p_r, self.prices[:,t], e)
            self.mean_indirect_utilities[:,t] = mean_indirect_util
            self.market_shares[:,t] = market_shares
            self.wages[:, t] = self.compute_wage(market_shares, self.prices[:,t], t)
            self.costs[:,t] = self.compute_marginal_cost(market_shares, t, self.wages[:,t])

            # Compute profits and markups for optimal prices
            self.compute_profit(t)
            self.compute_markup(t)

        self.compute_labor_from_quantity()

    def root_objective(self, price, v_p, e, t):

        """ The function is the first order condition of the 
        profif maximization problem of the firm

        Args:
            price (float): the price in each period - the value to be optimized
            v_p (type): random consumer specific demand shocks 
            e (type): the type 1 extreme value distributed error term 
            t (type): the time period for the optimization

        Returns:
            float: the profit first order condition that needs to be null
        """

        market_shares, all_probs, mean_indirect_utility = self.compute_share(v_p, price, e)
        wage = self.compute_wage(market_shares, price, t)
        cost = self.compute_marginal_cost(market_shares, t, wage)
        Jacobian = self.construct_Jacobian(all_probs, v_p)
        profit_FOC = np.matmul(np.transpose(Jacobian), (price - cost)) + market_shares
        return profit_FOC

    def compute_marginal_cost(self, market_shares, t, wage):
        """ Function to compute the marginal cost as a function of the 
        equilibiurm quantity produced in the market in each time period

        Args:
            market_shares (float): the market shares for that one particular period 
            t (int): the time period for which the cost has to be computed 

        Returns:
            float: the marginal cost for the time period t 
        """
        # This is the correct markginal cost 

        MC = (wage*(1/self.theta_L))*((self.n_consumers * market_shares)/(np.exp(self.theta_0 + self.productivity_shocks[:,t])*
              self.capital[:,t]**self.theta_K))**((1/self.theta_L)-1) *(1/(np.exp(self.theta_0 + self.productivity_shocks[:,t])*
              self.capital[:,t]**self.theta_K))
        
        # MC = np.ones(self.n_firms)
        return MC
    
    def compute_wage(self, market_shares, price, t): 
        """_summary_

        Args:
            market_shares (_type_): _description_
            price (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        derivative_revenue = (price*self.n_consumers*market_shares)*(
            (((self.n_consumers*market_shares)**(self.theta_L-1))*self.capital[:,t]**self.theta_K
             *np.exp(self.theta_0 + self.productivity_shocks[:,t])))
        wage = ((1/self.labor_elasticity)+1)/(derivative_revenue)

        return wage



    def compute_profit(self, t):
        """Funtion to compute profit 

        Args:
            t (int): the time period for which to compute the profit
        """
        self.profits[:,t] = self.market_shares[:,t]*(self.prices[:,t] - self.costs[:,t])


    def compute_markup(self, t):
        """Demand side markup - as in De Loeker p/c 

        Args:
            t (int): the period for which to compute the markup
        """
        self.markups[:,t] = self.prices[:,t]/self.costs[:,t]


    def compute_share(self, v_p, price, e):
        """Formula for computing the market share based on the multinomial 
        logit probability"

        Args:
            v_p (float): random consumer specific demand shock
            price (float): the optimal price in the fiven time period
            e (float): the type I extreme value error term

        Returns:
            float, float : The value of the market share and the individual 
            probabilities of purchasing a product
        """
        # The mean direct utility values


        price_r = np.reshape(price, (1, self.n_firms))
        alpha_0 = -np.exp(self.mu + (self.sigma)**2/2)

        beta = np.array([self.beta1, self.beta2, self.beta3])
        mean_indirect_utility = self.produc_chars@beta + alpha_0*price
        mean_indirect_utlity_for_utility = np.repeat(mean_indirect_utility, self.n_consumers, axis=0)

        alpha_i = np.reshape((-(np.exp(self.mu + self.sigma*v_p))+np.exp(self.mu + (self.sigma)**2/2)), (self.n_consumers, 1))
        random_coeff = np.ravel((alpha_i*price_r).T)

        u = mean_indirect_utlity_for_utility + random_coeff + e
        u_r = np.reshape(u, (self.n_firms, self.n_consumers))
        sum_u = np.sum(np.exp(u_r), axis =0)

        all_probs = np.exp(u_r)/(1 + sum_u)
        market_shares = np.sum(all_probs, axis=1)/self.n_consumers

        return market_shares, all_probs, mean_indirect_utility

    def construct_Jacobian(self, all_probs, v_p): 
        """ Formulas for the matrix of first order conditions of market
        shares with respect to prices 

        Args:
            v_p (float): random consumer demand shocks
            all_probs (float): the probability a consumer i buy a product j 

        Returns:
            float matrix : the Jacobian matrix of shares with respect to prices 

        updated version that should be giving the corrent result in the end 
        This needs to be further debugged because i think it doesn't some
        across correctly 
        """
        J = np.zeros((self.n_firms, self.n_firms))
        alphas = -np.exp(self.mu + self.sigma*v_p)
        for i in range(J.shape[0]):
            p1 = all_probs[i, :]
            for j in range(J.shape[1]):
                if i == j:
                    J[i, j] = np.sum(alphas * p1 - alphas * (p1 ** 2))/self.n_consumers
                else: 
                    p2 = all_probs[j, :]
                    J[i, j] = np.sum(-alphas * p1 * p2)/self.n_consumers
        return J

    def gen_product_chars(self):
        """Generates product characteristics"""
        X1 = np.random.uniform(self.x1_min, self.x1_max, size=self.n_firms)
        X2 = np.random.uniform(self.x2_min, self.x2_max, size=self.n_firms)
        all_X = np.column_stack((X1, X2))
        X0 = np.ones(self.n_firms)
        return np.column_stack((X0, all_X))
    
    def compute_investment(self, productivity, capital):
        """Generate optimal amount of investment for a given period 
        as a result of the capital and productivity evolution

        Args:
            productivity (float): the random productivity shock in that period 
            capital (floar): capital in that period

        Returns:
            float: investment level in the period
        """
        return (self.tau + self.gamma*productivity)*capital
    
    def capital_formation(self, capital, investment):
        """Capital formation over time 

        Args:
            capital (float): capital in the previous period
            investment (float): investment in the previous period

        Returns:
            float : capital in the next period 
        """
        return (1-self.tau)*capital + investment
    
        
    def compute_labor_from_quantity(self):
        """Computed contingent labor demand in the given market
        """
        self.labor_quantity = ((self.n_consumers* self.market_shares)/(self.capital**self.theta_K*
                                                              np.exp(self.theta_0 + self.productivity_shocks)
                                                              ))**(1/self.theta_L)
    


    def gen_productivity_capital_investments(self):
        """Generates the development of productivity, capital and investment 
        over time. As this values are independent from any optimal labor choice
        they can be generated together and initiated at the beginning as some 
        sort ofexogenous variables
        """

        # Initialization fo the martix
        productivity_shocks = np.zeros((self.n_firms, self.T))
        capital = np.zeros((self.n_firms, self.T))
        investments = np.zeros((self.n_firms, self.T))

        # Getting the first period initialization
        # There needs to be some sort of initialization for sigma and capital 
        productivity_shocks[:,0] = np.random.normal(self.mean_productivity, self.std_productivity, self.n_firms)
        capital[:, 0] = np.random.uniform(self.min_capital, self.max_capital, self.n_firms)

        investments[:, 0] = self.compute_investment(productivity_shocks[:, 0], capital[:, 0])

        for t in range(1, self.T):
            productivity_shocks[:, t] = (self.rho * productivity_shocks[:, t-1] 
                                         + np.random.normal(self.mean_productivity,
                                                             self.std_productivity, size=self.n_firms))
            capital[:, t] = self.capital_formation(capital[:,t-1], investments[:,t-1])
            investments[:, t] = self.compute_investment(productivity_shocks[:, t], capital[:, t])

        return productivity_shocks, capital, investments
    
    def save_simulation_data(self):
        """Saves dataframe with simulation data 
        """
        
        # This is such that data nicely stored
        time1 = np.reshape(np.repeat(np.array(range(1, self.T+1)), self.n_firms), (self.n_firms*self.T, 1))
        products1 = np.reshape(np.tile(np.array(range(1, self.n_firms+1)), self.T), (self.n_firms*self.T, 1))
        characteristic_1 = np.reshape(np.tile(self.produc_chars[:,1], self.T), (self.n_firms*self.T, 1))
        characteristic_2 = np.reshape(np.tile(self.produc_chars[:,2], self.T), (self.n_firms*self.T, 1))
        
        # All the data from the demand side 
        # prices1 = np.reshape(self.prices.T, (self.prices.size, 1))
        prices1 = self.prices.T.flatten()
        costs1 = self.costs.T.flatten()
        market_share1 = self.market_shares.T.flatten()
        mean_indirect_uti1 = self.mean_indirect_utilities.T.flatten()

        profits1 = self.profits.T.flatten()
        markups1 = self.markups.T.flatten()

        # Quantity/market equilibiurm data
        quantity = self.n_consumers * market_share1

        # All the data from the supply side 
        capital1 =  self.capital.T.flatten()
        investment1 =  self.investments.T.flatten()
        productivity1 = self.productivity_shocks.T.flatten()
        labor1_quantity = self.labor_quantity.T.flatten()
        wages1 = self.wages.T.flatten()

        # Generate the dataframe with all the information
        df_simulation = pd.DataFrame({'market_ids': time1.T[0],
                                    'firm_ids':products1.T[0],
                                    'characteristic1':characteristic_1.T[0], 
                                    'characteristic2':characteristic_2.T[0], 
                                    'prices':prices1, 
                                    'marginal_cost':costs1,
                                    'shares':market_share1, 
                                    'profits':profits1, 
                                    'markups':markups1,
                                    'e_quantity':quantity,
                                    'capital':capital1,
                                    'investment':investment1,
                                    'productivity':productivity1,
                                    'labor':labor1_quantity, 
                                    'indirect_util':mean_indirect_uti1,
                                    'wage':wages1
                                    })
        df_simulation.to_csv(f'data/data_endogenouswage/market_endogenouswage_{self.seed}.csv', index=False)
        print(df_simulation)
