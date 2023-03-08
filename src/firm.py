from numpy import array
from scipy import optimize
import matplotlib.pyplot as plt


# class Firm:
#     def __init__(self, id, shares, cost):
        
#         self.id = id
#         self.shares = shares
#         self.cost = cost 

# c_s = 1
# c_t = 3

# q_s = 100 - 5*(p_s - p_t)
# q_t = 100 - 5*(p_t - p_s)

# profit_s = (p_s - c_s)*100 - 5*(p_s - p_t)
# proft_t = (p_t - c_t)*q_t


# def func(p, c, p_c):
#    return ((p - c)*(100 - 5*(p - p_c)))

# p = np.array[]

# c1 = np.array([10,12.])
# c2 = np.array([3, 5.])
# optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))
# array([ 1.4920333 ,  1.37228132


def demand(x1,x2,x3,b):
    return 100- b*(x1 - x2- x3)



def profit(x1,x2, x3,c1,b):
    return demand(x1,x2,x3, b)*(x1 - c1)



def reaction(x3, x2,c1,b):
    x1 = optimize.brute(lambda x: -profit(x,x3, x2,c1,b), ((0,1,),)) # brute minimizes the function;
                                                                 # when we minimize -profits, we maximize profits
    return x1[0]


def vector_reaction(x,param): # vector param = (b,c1,c2)
    return array(x)-array([reaction(x[2], x[1],param[1],param[0]),reaction(x[0], x[2],param[2],param[0]), 
                           reaction(x[0], x[1],param[3],param[0])])


param = [5, 1, 3, 2]
# initial guess x0
x0 = [1, 1, 1]

ans = optimize.fsolve(vector_reaction, x0, args = (param))
print(ans)



