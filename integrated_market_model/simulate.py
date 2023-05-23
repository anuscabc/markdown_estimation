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
    n_consumers = 500
    n_chars = 1
    T = 100
    # s = 400
    for s in range(1, 10, 1):
        single_simulation(n_firms, n_consumers, n_chars, T, s)

