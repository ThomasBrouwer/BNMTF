"""
Class representing an exponential distribution, allowing us to sample from it.
"""
from numpy.random import exponential

# Exponential draws
def exponential_draw(lambdax):
    scale = 1.0 / lambdax
    return exponential(scale=scale,size=None)
        
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
scale = 2.
s = [exponential_draw(1./scale) for i in range(0,1000)] 
s2 = np.random.exponential(scale, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
plt.show()
'''