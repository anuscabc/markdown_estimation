cls 
clear all 

//BLP estimation with the generated data 


global path "/Users/popovici/Desktop/Thesis/markdown_estimation/data"
cd $path
import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_4.csv"
describe

blp shares characteristic1 characteristic2, endog(prices = labor, capital) stochastic(prices) markets(market_ids) draws(1000) 

// estimates save estimates_1_stata.csv, append 


// import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_400.csv", clear
//
// blp shares characteristic1 characteristic2, endog(prices = labor, capital) stochastic(prices) markets(market_ids) draws(1000) 
//
//

reg prices labor capital

