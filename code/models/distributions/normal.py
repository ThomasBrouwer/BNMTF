"""
Class representing an normal distribution, allowing us to sample from it.
"""
from numpy.random import normal
import numpy, math

# Draw a value for tau ~ Gamma(alpha,beta)
def normal_draw(mu,tau):
    sigma = numpy.float64(1.0) / math.sqrt(tau)
    return normal(loc=mu,scale=sigma,size=None)
    
       
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
mu = -1.
tau = 4.
sigma = 1./2.
s = [normal_draw(mu,tau) for i in range(0,1000)] 
s2 = np.random.normal(mu,sigma, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
plt.show()
'''