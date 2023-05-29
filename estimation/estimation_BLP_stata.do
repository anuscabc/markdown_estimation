cls 
clear all 

//BLP estimation with the generated data 
global path "/Users/popovici/Desktop/Thesis/markdown_estimation/data"
cd $path
import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_4.csv"
describe


drop if shares<0.00001
drop if shares == 1

egen sum_shares = sum(shares), by(market_ids) 
gen outside_good = 1 -sum_shares

gen y = ln(shares) - ln(outside_good)

reg y characteristic1 characteristic2 price, robust 

ivregress 2sls y characteristic1 characteristic2 (price=labor), robust

blp shares characteristic1 characteristic2, endog(prices= marginal_cost, productivity) stochastic(prices) markets(market_ids) draws(500) 




