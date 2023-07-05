cls 
clear all 


foreach i of numlist 1/1000{
	
//BLP estimation with the generated data 
global path "/Users/popovici/Desktop/Thesis/markdown_estimation/data"
cd $path

import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_`i'.csv"


drop if shares<0.00001
drop if shares == 1
drop if shares > 0.99
gen market_size = 10000


// make the variable log terms 
gen lny = log(e_quantity)
gen lnl = log(labor)
gen lnk = log(capital)
gen lni = log(investment)
// generate additional terms for the polynomial
gen double lninvlnkop = lni*lnk
gen double lninvsq = lni^2
gen double lnkopsq = lnk^2
gen double lninvcube = lni^3
gen double lnkopcube = lnk^3



// USE THE OLS ESITMATION PROCEDURE 
qui reg lny lnl lnk
estimates store e1
mat try_ols = r(table)
gen bl_ols = try_ols[1, 1]
gen bk_ols = try_ols[1, 2]


xtset firm_ids market_ids
// The within estimator
qui xtreg lny lnl lnk
mat try_xtreg= r(table)
gen bl_xtreg = try_xtreg[1, 1]
gen bk_xtreb = try_xtreg[1, 2]


// The ACF cause i do not have firm exist and dropout as predictivitlity 
qui acfest lny, free(lnl) state(lnk) proxy(lni) i(firm_ids) t(market_ids) nodum invest

mat try_acfest=r(table)
gen bk_acfest = try_acfest[1, 1]
gen bl_acfest = try_acfest[1, 2]


//OP 
// Generate some of the values needed for the polynomial 
// Adapting the Olley and Pakes appendix (cite)
// Step I - regress lny on variable inputs and
// polynomial in i, k
qui regress lny lnl lni lnk lninvlnkop lninvsq lnkopsq lninvcube lnkopcube 
mat try_op =r(table)
gen bl_op = try_op[1, 1]


// To get the capital coefficients even if not needed on this case 
// making csv file with only the estimates 
keep bl_* bk_*
drop if _n>1

export delimited using "values_estimation_production_`i'.csv", replace 
clear all
}
