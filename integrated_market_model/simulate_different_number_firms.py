from market import IntegratedMarketModel
import numpy as np 
import matplotlib.pyplot as plt 


mean_prices = []
mean_share = []

def single_simulation(n_firms, n_consumers, n_chars, T, s):
    model = IntegratedMarketModel(
        n_firms, 
        n_consumers, 
        n_chars, 
        T, 
        seed=s
    )
    model.demand_side_optimisation()
    # print to see what it 
    # looks like before saving the flatten array and taking the mean over the different 
    # number of firms 
    flatten_prices = model.prices.T.flatten()
    cum_market_share = 1 -  np.sum(model.market_shares, axis=0)
    return flatten_prices.mean(), cum_market_share





if __name__ == "__main__":

    # n_firms = 2
    n_consumers = 500
    n_chars = 2
    T = 1
    s = 200
    for n_firms in range(1, 20, 1):
        res_prices, res_mm = single_simulation(n_firms, n_consumers, n_chars, T, s)
        mean_prices.append(res_prices)
        mean_share.append(res_mm)
    print(mean_prices)
    print(mean_share)

    fig, axs = plt.subplots(2, figsize=(6, 6))
    fig.suptitle(f'Market Equilibium Over Different Number of Firms')
    axs[0].plot(mean_prices, color='orange')
    axs[0].set_title("Price")
    axs[1].plot(mean_share, color='green')
    axs[1].set_title("Outside Good Share")

    fig.tight_layout()

    fig.savefig("plots/diff_firms.png")





