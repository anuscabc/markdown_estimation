import matplotlib.pyplot as plt 


year = [2019, 2020, 2021, 2022, 2023]

numbers = [29800, 22300, 38000, 64894, 100000]


plt.plot(year, numbers)
plt.xlabel('Year')
plt.xticks(year)
plt.ylabel('Non-EU Workers (2023 expected)')
plt.show()