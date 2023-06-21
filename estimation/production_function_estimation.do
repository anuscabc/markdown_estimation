cls 
clear all 


import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_4.csv"


drop if shares<0.00001
drop if shares == 1


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


xtset firm_ids market_ids
sort firm_ids market_ids


// USE THE OLS ESITMATION PROCEDURE 
reg lny lnl lnk
estimates store e1


// The within estimator
xtreg lny lnl lnk
estimates store e2


// The ACF cause i do not have firm exist and dropout as predictivitlity 
acfest lny, free(lnl) state(lnk) proxy(lni) i(firm_ids) t(market_ids) 
predict omega_hat, omega 
histogram omega_hat



//OP 
// Generate some of the values needed for the polynomial 
// Adapting the Olley and Pakes appendix (cite)
// Step I - regress lny on variable inputs and
// polynomial in i, k
regress lny lnl lni lnk lninvlnkop lninvsq lnkopsq lninvcube lnkopcube 

predict double lny_hat if e(sample), xb
scalar b_lnl = _b[lnl]

/// Step III -- Nonlinear regression of y - lnl*b_lnl - lnm*b_lnm// on age, capital, and the polynomial to control for selection

// Output minus the contributions of the variable inputs
generate double lhs = lny - lnl*b_lnl




// Finally, fit the linear model (in the case of no firm exit)
nl (lhs = b0 + bklnk+ )






