"""
Class representing an exponential distribution, allowing us to sample from it.
"""
from numpy.random import exponential

class Exponential:
    def __init__(self,lambdax):
        self.lambdax = float(lambdax)
        
    # Draw a value for tau ~ Gamma(alpha,beta)
    def draw(self):
        scale = 1.0 / self.lambdax
        return exponential(scale=scale,size=None)
        
        
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
scale = 2.
expdist = Exponential(1.0/scale)
s = [expdist.draw() for i in range(0,1000)] 
s2 = np.random.exponential(scale, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
plt.show()
'''