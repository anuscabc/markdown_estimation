cls 
clear all 

//BLP estimation with the generated data 
global path "/Users/popovici/Desktop/Thesis/markdown_estimation/data"
cd $path
import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates.csv"
describe


// for i in numlist 1/9{
// 	import delimited 	"/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_`i'.csv"
//	
// 	egen sum_shares = sum(shares), by(market_ids) 
// 	gen outside_good = 1 -sum_shares
//
// 	gen y = ln(shares) - ln(outside_good)
//
//
//
// 	reg y characteristic1 characteristic2 price, robust 
// 	estimates store e1 
// 	# write output to csv file 
// 	constant = _cons[0] 
//
// 	ivregress 2sls y characteristic1 characteristic2 (price= capital labor), robust
// 	estimates store e2
//
// 	blp shares characteristic1 characteristic2, endog(prices= capital labor) stochastic(prices) markets(market_ids) draws(1000)
// 	estimates store e3
//
//	
//	
// }

//
// drop if shares<0.00001
// drop if shares == 1

egen sum_shares = sum(shares), by(market_ids) 
gen outside_good = 1 -sum_shares

gen y = ln(shares) - ln(outside_good)



reg y characteristic1 characteristic2 price, robust 
estimates store e1 

ivregress 2sls y characteristic1 characteristic2 (price= capital labor), robust
estimates store e2

blp shares characteristic1 characteristic2, endog(prices= capital labor) stochastic(prices) markets(market_ids) draws(1000)
estimates store e3


esttab e* using blp.tex,  se r2  b(3) se(2) sfmt(2ß) obslast star(* 0.10 ** 0.05 *** 0.01) replace  //getting latex table

