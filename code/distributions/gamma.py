"""
Class representing a gamma distribution, allowing us to sample from it, 
and compute the expectation and the expectation of the log.
"""
import math
from scipy.special import psi as digamma
from numpy.random import gamma

class Gamma:
    def __init__(self,alpha,beta):
        self.alpha = float(alpha)
        self.beta = float(beta)
        
    # Draw a value for tau ~ Gamma(alpha,beta)
    def draw(self):
        shape = self.alpha
        scale = 1.0 / self.beta
        return gamma(shape=shape,scale=scale,size=None)
        
    # Return expectation, E[tau] = alpha / beta
    def expectation(self):
        return self.alpha / self.beta
        
    # Return expectation of log_e, E[log tau] = digamma(alpha) - log beta
    def expectation_log(self):
        return digamma(self.alpha) - math.log(self.beta)
        
        
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import scipy.special as sps
import numpy as np
shape, scale = 2., 2. # mean and dispersion
gammadist = Gamma(shape,1.0/scale)
s = [gammadist.draw() for i in range(0,1000)] 
s2 = np.random.gamma(shape, scale, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                     (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()
'''