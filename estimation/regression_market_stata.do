cls 
clear all 

//leave it local for the moment 
import delimited "/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_100.csv"

// Getting the share of the outside good for each market 

bys market_ids: sum()

xtset firm_ids market_ids



gen y = ln(shares) - ln

xtreg 
