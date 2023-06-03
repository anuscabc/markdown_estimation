from market import IntegratedMarketModel


mean_prices = []

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
    return flatten_prices.mean()




if __name__ == "__main__":

    # n_firms = 2
    n_consumers = 500
    n_chars = 1
    T = 1
    s = 123
    for n_firms in range(1, 15, 1):
        res = single_simulation(n_firms, n_consumers, n_chars, T, s)
        mean_prices.append(res)
    print(mean_prices)
