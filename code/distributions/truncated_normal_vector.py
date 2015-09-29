"""
Class representing a Truncated Normal distribution, with a=0 and b-> inf, 
allowing us to sample from it, and compute the expectation and the variance.

This is the special case that we want to compute the expectation and variance
for multiple independent variables - i.e. mu and tau are vectors.

truncnorm: a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
           loc, scale = mu, sigma
           
We get efficient draws using the library rtnorm by C. Lassner, from:
    http://miv.u-strasbg.fr/mazet/rtnorm/
We compute the expectation and variance ourselves - note that we use the
complementary error function for 1-cdf(x) = 0.5*erfc(x/sqrt(2)), as for large
x (>8), cdf(x)=1., so we get 0. instead of something like n*e^-n.

As mu gets lower (negative), and tau higher, we get draws and expectations that
are closer to an exponential distribution with scale parameter mu * tau.

The draws in this case work effectively, but computing the mean and variance
fails due to numerical errors. As a result, the mean and variance go to 0 after
a certain point.
This point is: -38 * std.

This means that we need to use the mean and variance of an exponential when
|mu| gets close to 38*std.
Therefore we use it when |mu| < 30*std.
"""
import math, numpy, time
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
from scipy.special import erfc
import rtnorm


# TN draws
def TN_vector_draw(mus,taus):
    sigmas = numpy.float64(1.0) / numpy.sqrt(taus)
    draws = []
    for (mu,sigma,tau) in zip(mus,sigmas,taus):
        if tau == 0.:
            draws.append(0)
        else:
            d = rtnorm.rtnorm(a=0., b=numpy.inf, mu=mu, sigma=sigma)[0]
            d = d if (d >= 0. and d != numpy.inf and d != -numpy.inf and not numpy.isnan(d)) else 0.
            draws.append(d)   
    '''
    draws = parallel_draw(self.mu,self.sigma,self.tau)
    '''     
    return draws           
       
# TN expectation    
def TN_vector_expectation(mus,taus):
    sigmas = numpy.float64(1.0) / numpy.sqrt(taus)
    x = - numpy.float64(mus) / sigmas
    lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
    exp = mus + sigmas * lambdax
    
    # Exp expectation - overwrite value if mu < -30*sigma
    exp = [1./(numpy.abs(mu)*tau) if mu < -30 * sigma else v for v,mu,tau,sigma in zip(exp,mus,taus,sigmas)]
    return [v if (v >= 0.0 and v != numpy.inf and v != -numpy.inf and not numpy.isnan(v)) else 0. for v in exp]
    
# TN variance
def TN_vector_variance(mus,taus):
    sigmas = numpy.float64(1.0) / numpy.sqrt(taus)
    x = - numpy.float64(mus) / sigmas
    lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
    deltax = lambdax*(lambdax-x)
    var = sigmas**2 * ( 1 - deltax )
    
    # Exp variance - overwrite value if mu < -30*sigma
    var = [(1./(numpy.abs(mu)*tau))**2 if mu < -30 * sigma else v for v,mu,tau,sigma in zip(var,mus,taus,sigmas)]
    return [v if (v >= 0.0 and v != numpy.inf and v != -numpy.inf and not numpy.isnan(v)) else 0. for v in var]      
       
# TN mode
def TN_vector_mode(mus):
    zeros = numpy.zeros(len(mus))
    return numpy.maximum(zeros,mus)   
       

""" Methods for parallel draws """
'''
from joblib import Parallel, delayed
import itertools
import multiprocessing

def draw(mu,sigma,tau):
    if tau == 0.:
        return 0
    else:
        d = rtnorm.rtnorm(a=0., b=numpy.inf, mu=mu, sigma=sigma)[0]
        d = d if (d >= 0. and d != numpy.inf and d != -numpy.inf and not numpy.isnan(d)) else 0.
        return d
        
def parallel_draw(mus,sigmas,taus):
    draws = Parallel(n_jobs=-1)( #-1 = no. of machine cores
        delayed(draw)(mu,sigma,tau) for (mu,sigma,tau) in itertools.izip(mus,sigmas,taus)
    )
    return draws
'''