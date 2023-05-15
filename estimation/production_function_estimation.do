cls 
clear all 


import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_1.csv"

// make the variable log terms 
gen lny = log(e_quantity)
gen lnl = log(labor)
gen lnk = log(capital)
gen lni = log(investment)


xtset firm_ids market_ids
sort firm_ids market_ids


// OLLEY AND PAKES ESTIMATION DOES NOT WORK BECUASE NO EXIT IN THE MODEL 
// opreg lny, exit(exit_var) state(lnk) proxy(lni) free(lnl) vce(bootstrap)



// USE THE ACF ESITMATION PROCEDURE 
reg lny lnl lnk

xtreg lny lnl lnk

acfest lny, free(lnl) state(lnk) proxy(lni) i(firm_ids) t(market_ids) 


