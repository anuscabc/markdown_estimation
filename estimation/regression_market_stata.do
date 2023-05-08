cls 
clear all 

//leave it local for the moment 
import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_100.csv"

// Getting the share of the outside good for each market 

egen total_share = sum(shares), by(market_ids) 
gen outside_good = 1 - total_share
gen y = shares - outside_good


reg y prices characteristic1 


ivregress 2sls y characteristic1 characteristic2 (price = marginal_cost characteristic1 characteristic2 )


reg prices marginal_cost characteristic1 characteristic2
predict predicted_prices


reg y predicted_prices characteristic1 characteristic2

