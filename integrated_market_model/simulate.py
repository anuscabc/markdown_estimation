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

    n_firms = 4
    n_consumers = 1000
    n_chars = 2
    T = 100
    s = 300
    # for s in range(1, 10, 1):
    #     single_simulation(n_firms, n_consumers, n_chars, T, s)
    single_simulation(n_firms, n_consumers, n_chars, T, s)
