from market import IntegratedMarketModel
import numpy as np 
import matplotlib.pyplot as plt 


mean_prices = []
mean_share = []
mean_cost = []
mean_markups = []

def single_simulation(n_firms, n_consumers, n_chars, T, s, x1_min, x2_min):
    model = IntegratedMarketModel(
        n_firms, 
        n_consumers, 
        n_chars, 
        T, 
        x1_min=x1_min,
        x2_min=x2_min,
        seed=s
    )
    model.demand_side_optimisation()
    # print to see what it 
    # looks like before saving the flatten array and taking the mean over the different 
    # number of firms 
    flatten_prices = model.prices.T.flatten()
    flatten_cost = model.costs.T.flatten()
    flatten_markups = model.markups.flatten()
    cum_market_share = 1 -  np.sum(model.market_shares, axis=0)
    return flatten_prices.mean(), cum_market_share, flatten_cost.mean(), flatten_markups.mean()





if __name__ == "__main__":

    """This function now simlates the model for diffeent number of product characteritics and looks how 
    prices, market shares, costs and markups react -> 
    """

    n_firms = 10
    n_consumers = 1000
    n_chars = 2
    T = 1
    s = 200
    for x_min in range(1, 5, 1):
        res_prices, res_mm, res_c, res_mk = single_simulation(n_firms, n_consumers, n_chars, T, s, x1_min=x_min, x2_min=x_min)
        mean_prices.append(res_prices)
        mean_share.append(res_mm)
        mean_cost.append(res_c)
        mean_markups.append(res_mk)
    print(mean_prices)
    print(mean_share)

    fig, axs = plt.subplots(4, figsize=(6, 6))
    fig.suptitle(f'Market Equilibium Over Less differentiation product characteristics')
    axs[0].plot(mean_prices, color='orange')
    axs[0].set_title("Price")
    axs[1].plot(mean_share, color='green')
    axs[1].set_title("Outside Good Share")
    axs[2].plot(mean_cost, color='blue')
    axs[2].set_title("Mean Cost")
    axs[3].plot(mean_cost, color='pink')
    axs[3].set_title("Mean Markups")

    fig.tight_layout()
    fig.savefig("plots/diff_firms.png")





