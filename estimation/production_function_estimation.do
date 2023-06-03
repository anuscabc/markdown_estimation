cls 
clear all 


import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_1.csv"

gen exit_var=0 if firm==2 & market_ids>=80
replace exit_var = 1 

// make the variable log terms 
gen lny = log(e_quantity)
gen lnl = log(labor)
gen lnk = log(capital)
gen lni = log(investment)


xtset firm_ids market_ids
sort firm_ids market_ids


// USE THE ACF ESITMATION PROCEDURE 
reg lny lnl lnk
estimates store e1



// The within estimator
xtreg lny lnl lnk
estimates store e2


// The ACF cause i do not have firm exist and dropout as predictivitlity 
acfest lny, free(lnl) state(lnk) proxy(lni) i(firm_ids) t(market_ids) 



esttab e* using productiv.tex,  se r2  b(3) se(2) sfmt(2) obslast star(* 0.10 ** 0.05 *** 0.01) replace  //getting latex table




// Adapting the Olley and Pakes appendix 

// OLLEY AND PAKES ESTIMATION DOES NOT WORK BECUASE NO EXIT IN THE MODEL 
opreg lny, exit(exit_var) state(lnk) proxy(lni) free(lnl)
