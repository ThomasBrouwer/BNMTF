"""
Class representing a gamma distribution, allowing us to sample from it, 
and compute the expectation and the expectation of the log.
"""
import math
from scipy.special import psi as digamma
from numpy.random import gamma


# Gamma draws
def gamma_draw(alpha,beta):       
    shape = float(alpha)
    scale = 1.0 / float(beta)
    return gamma(shape=shape,scale=scale,size=None)
        
# Gamma expectation
def gamma_expectation(alpha,beta): 
    alpha, beta = float(alpha), float(beta)      
    return alpha / beta
        
# Gamma variance
def gamma_expectation_log(alpha,beta):   
    alpha, beta = float(alpha), float(beta)      
    return digamma(alpha) - math.log(beta)
   
# Gamma mode
def gamma_mode(alpha,beta):
    alpha, beta = float(alpha), float(beta)
    return (alpha-1) / beta


'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import scipy.special as sps
import numpy as np
shape, scale = 2., 2. # mean and dispersion
s = [gamma_draw(shape,1.0/scale) for i in range(0,1000)] 
s2 = np.random.gamma(shape, scale, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                     (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()
'''