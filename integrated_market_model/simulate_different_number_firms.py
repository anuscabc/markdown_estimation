from market import IntegratedMarketModel
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


mean_price_car = []
mean_share_car = []
mean_cost_car = []
mean_markups_car = []

mean_price_firms = []
mean_share_firms = []
mean_cost_firms = []
mean_markups_firms = []



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


def make_graphs_diff_products(mean_price_car, mean_share_car, mean_cost_car, mean_markups_car):
    fig, axs = plt.subplots(4, figsize=(6, 6))
    fig.suptitle(f'Market Equilibium Over Less Differentiation in \n Product Characteristics and Marginal Constant Cost')
    axs[0].plot(mean_price_car, color='#FFA3B9')
    axs[0].set_title("Price")
    axs[1].plot(mean_share_car, color='#FFC580')
    axs[1].set_title("Outside Good Share")
    axs[2].plot(mean_cost_car, color='blue')
    axs[2].set_title("Mean Cost")
    axs[3].plot(mean_markups_car, color= '#C994FF')
    axs[3].set_title("Mean Markups")

    fig.tight_layout()
    fig.savefig("plots/diff_prod.png")


def make_graphs_diff_firms(mean_price_firm, mean_share_firm, mean_cost_firm, mean_markups_firm):
    fig, axs = plt.subplots(4, figsize=(6, 6))
    fig.suptitle(f'Market Equilibium Over Different Number of Firms and Constant Marginal Cost')
    axs[0].plot(mean_price_firm, color='#FFA3B9')
    axs[0].set_title("Price")
    axs[1].plot(mean_share_firm, color='#FFC580')
    axs[1].set_title("Outside Good Share")
    axs[2].plot(mean_cost_firm, color='blue')
    axs[2].set_title("Mean Cost")
    axs[3].plot(mean_markups_firm, color= '#C994FF')
    axs[3].set_title("Mean Markups")

    fig.tight_layout()
    fig.savefig("plots/diff_firms.png")





if __name__ == "__main__":

    """This function now simlates the model for diffeent number of product characteritics and looks how 
    prices, market shares, costs and markups react -> 

    The same over different number of firms in the market for the same characteristics

    """

    n_firms = 10
    n_consumers = 1000
    n_chars = 2
    T = 1
    s = 200
    x_min=3

    for x_min in range(1, 5, 1):
        res_prices, res_mm, res_c, res_mk = single_simulation(n_firms, n_consumers, n_chars, T, s, x1_min=x_min, x2_min=x_min)
        mean_price_car.append(res_prices)
        mean_share_car.append(res_mm)
        mean_cost_car.append(res_c)
        mean_markups_car.append(res_mk)

    make_graphs_diff_products(mean_price_car, mean_share_car, mean_cost_car, mean_markups_car)

    for n_firms in range(1, 30, 1):
        res_prices, res_mm, res_c, res_mk = single_simulation(n_firms, n_consumers, n_chars, T, s, x1_min=x_min, x2_min=x_min)
        mean_price_firms.append(res_prices)
        mean_share_firms.append(res_mm)
        mean_cost_firms.append(res_c)
        mean_markups_firms.append(res_mk)


    make_graphs_diff_firms(mean_price_firms, mean_share_firms, mean_cost_firms, mean_markups_firms)
    print(mean_markups_firms)
    print(mean_share_firms)







