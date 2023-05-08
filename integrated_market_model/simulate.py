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

    n_firms = 20
    n_consumers = 1000
    n_chars = 2
    T = 100
    single_simulation(n_firms, n_consumers, n_chars, T)
