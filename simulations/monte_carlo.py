# Monte Carlo simulation for the approximation og the market shares 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import firm_revised
# Get in the dataframe that you want and then unpack everything from it for the monte carlo 


# Need to put all the parametrs in the same vextor 
def f(theta, df_MC, df):
    shares_true = df['shares'].to_numpy()
    shares_est, _ = firm_revised.share()
    return np.linalg.norm(shares_true - shares_est.to_numpy())

res = minimize(f, theta_0, args=(df_MC, df), method = 'Nelder-Mead')
print(res.x)
print(theta_true)

end = time.time()

print("The time of execution of above program is :",
      (end-start) /60, "minutes")
 