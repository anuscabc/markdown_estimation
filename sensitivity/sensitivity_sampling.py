"""
Generates samples for sensitivity analysis

"""

import numpy as np
import pandas as pd
from SALib.sample import sobol
import sys

sys.path.append('integrated_market_model')
from market import IntegratedMarketModel


def generate_parameters(
        sampling_method:str, 
        seed:int, 
        problem:dict, 
        param_names:np.ndarray, 
        param_bounds:np.ndarray
    ):
    """_summary_

    Args:
        sampling_method (str): _description_
        seed (int): _description_
        problem (dict): _description_
        param_names (np.ndarray): _description_
        param_bounds (np.ndarray): _description_
    """
    
    # TODO change method to match given sampling method
    N = 5 * 2 ** len(param_names)
    param_values = sobol.sample(problem, N=N, calc_second_order=True, seed=seed)

    df = pd.DataFrame(param_values, columns=param_names)
    df.to_csv(f'data/{sampling_method}_params.csv', index=False)


def generate_samples(sampling_method:str, n_firms:int, n_consumers:int, n_chars:int, T:int, output_names):
    """_summary_

    Args:
        sampling_method (str): _description_
        n_firms (int): _description_
        n_consumers (int): _description_
        n_chars (int): _description_
        T (int): _description_
        output_names (_type_): _description_
    """

    outputs = []

    df_in = pd.read_csv(f'data/{sampling_method}_params.csv')
    param_names = df_in.columns.to_numpy()
    
    for idx, param_values in df_in.iterrows():

        print(f"simulation {idx}")
        
        # Make model with default parameters
        model = IntegratedMarketModel(n_firms, n_consumers, n_chars, T)

        # Change parameters used for SA
        for param_name, param_value in zip(param_names, param_values):
            setattr(model, param_name, param_value)

        # Run model and save output
        model.demand_side_optimisation()

        # TODO save relevant output
        min_mm, max_mm = model.compute_extr_cum_marketshare()
        min_p, max_p = model.compute_extr_cum_prices()
        min_l, max_l = model.compute_extr_cum_labor()
        mean_mm = model.compute_mean_marketshare() 
        mean_p = model.compute_mean_price()
        mean_l = model.compute_mean_labor()
        run_output = [mean_mm, mean_p, mean_l, min_mm, max_mm, min_p, max_p, min_l, max_l]
        outputs.append(run_output)

        # if idx==1: 
        #     break

    
    output = np.array(outputs)
    df_out = pd.DataFrame(output, columns=output_names)
    df_out.to_csv(f'data/{sampling_method}_output.csv', index=False)

   

if __name__ == "__main__":

    n_firms = 10
    n_consumers = 500
    n_chars = 2
    T = 1
    seed = 1234

    sampling_method = 'sobol'

    # Define parameters of interest
# Define parameters of interest
# Define parameters of interest
    # param_names = ['beta1', 'beta2', 'beta3', 'mu', 'omega']
    # param_bounds = [[2., 3.], [-0.4, -0.1], [-0.4, -0.1], [0.1, 0.6], [0.1, 0.5]]
    param_names = ['beta1', 'beta2', 'beta3']
    param_bounds = [[1., 3.], [-0.5, -0.3], [-0.5, -0.3]]
    problem = {
    'num_vars': len(param_names),
    'names': param_names,
    'bounds': param_bounds
    } 
    # Define output parameters of interest
    # output_names = ['min_mm', 'max_mm', 'min_p', 'max_p', 'min_l', 'max_l']
    output_names = ['mean_mm', 'mean_p', 'mean_l', 'min_mm', 'max_mm', 'min_p', 'max_p', 'min_l', 'max_l']

    generate_parameters(sampling_method, seed, problem, param_names, param_bounds)
    generate_samples(sampling_method, n_firms, n_consumers, n_chars, T, output_names)