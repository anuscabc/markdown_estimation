cls 
clear all 




foreach i of numlist 1/99{
	
//BLP estimation with the generated data 
global path "/Users/popovici/Desktop/Thesis/markdown_estimation/data"
cd $path

import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_`i'.csv"


drop if shares<0.00001
drop if shares == 1
gen market_size = 10000

bys market_id: gen shares1 = e_quantity/market_size 


egen sum_shares = sum(shares1), by(market_ids) 
gen outside_good = 1 -sum_shares

gen y = ln(shares1) - ln(outside_good)


qui reg y characteristic1 characteristic2 price, robust 
mat try_ols = r(table)
gen b2_ols = try_ols[1, 1]
gen b3_ols = try_ols[1, 2]
gen a_ols = try_ols[1, 3]
gen b1_ols = try_ols[1, 4]





qui ivregress 2sls y characteristic1 characteristic2 (price= capital labor), robust
estimates store e2
mat try0 = r(table)
gen a_iv = try0[1, 1]
gen b2_iv = try0[1, 2]
gen b3_iv = try0[1, 3]
gen b1_iv = try0[1, 4]



qui blp shares characteristic1 characteristic2, endog(prices= capital labor) stochastic(prices) markets(market_ids) draws(1000)
mat try1 = r(table)
gen b1_blp = try1[1, 1]
gen b2_blp = try1[1, 2]
gen b3_blp = try1[1, 3]
gen a_blp = try1[1, 4]
gen sigma_blp = try1[1, 5]
// estimates store e3




qui rcl shares1 characteristic1 characteristic2 (prices = capital labor), market(market_ids) rc(prices) draws(1000) msize(market_size)
return list
mat try2 =  e(b)   
gen sigma_rcl = try2[1, 1]
gen a_rcl = try2[1, 2]
gen b2_rcl = try2[1, 3]
gen b3_rcl = try2[1, 4]
gen b1_rcl = try2[1, 5]


// making csv file with only the estimates 
keep a_* b1_* b2_* b3_* sigma_*
drop if _n>1

export delimited using "values_estimation_blp_`i'.csv", replace 
clear all

}


