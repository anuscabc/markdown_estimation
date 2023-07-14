from endogenouswage_market import EndogenousWageMarketModel

def single_simulation(n_firms, n_consumers, n_chars, T, s):
    model = EndogenousWageMarketModel(
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
    s = 4
    single_simulation(n_firms, n_consumers, n_chars, T, s)

