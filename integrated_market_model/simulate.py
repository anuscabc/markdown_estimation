from market import IntegratedMarketModel

def single_simulation(n_firms, n_consumers, n_chars, T):
    model = IntegratedMarketModel(
        n_firms, 
        n_consumers, 
        n_chars, 
        T
    )
    model.demand_side_optimisation()
    model.save_simulation_data()

if __name__ == "__main__":

    n_firms = 10
    n_consumers = 500
    n_chars = 2
    T = 2
    single_simulation(n_firms, n_consumers, n_chars, T)
