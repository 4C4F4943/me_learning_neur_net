import matplotlib.pyplot as plt 
from statistics import mean
import numpy as np
import random
import time
############## this is a function that calculates the best fit slope and then plots it at the end #############
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m * mean(xs)
    return m,b

idk_x = np.array([1,4,3,6])
idk_y = np.array([5,6,8,10])
random.shuffle(idk_x)
random.shuffle(idk_y)
m,b = best_fit_slope(idk_x,idk_y)
plt.plot(idk_x,idk_y,color="blue")
plt.plot(idk_x, idk_x*m+b)
plt.show()