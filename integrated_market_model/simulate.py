from market import IntegratedMarketModel

def single_simulation(n_firms, n_consumers, n_chars, T, s):
    model = IntegratedMarketModel(
        n_firms, 
        n_consumers, 
        n_chars, 
        T, 
        seed=s
    )
    model.demand_side_optimisation()
    model.save_simulation_data()




if __name__ == "__main__":

    n_firms = 10
    n_consumers = 10000
    n_chars = 2
    T = 100
    # s = 100
    for s in range(101, 1001, 1):
        single_simulation(n_firms, n_consumers, n_chars, T, s)

