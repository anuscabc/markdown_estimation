library(AER)
library(plm)
library(stargazer)

data <- read.csv("/Users/popovici/Desktop/Thesis/markdown_estimation/data/market_integrates_1.csv")

# Generating the needed log values in df 

data['ln_y'] = log(data['e_quantity'])
data['ln_l'] = log(data['labor'])
data['ln_k'] = log(data['capital'])

reg1 <- lm(formula = ln_y ~ ln_l + ln_k, data = data)

print(reg1)