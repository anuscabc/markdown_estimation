cls
clear all 

cd "/Users/popovici/Desktop/Thesis/markdown_estimation/estimation_trial1"


import excel "/Users/popovici/Desktop/Thesis/markdown_estimation/data/data_clean.xlsx", sheet("Sheet1") firstrow clear

drop A Unnamed0

// Get some sort of summary of the simulation data 
