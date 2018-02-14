"""
Class representing a Truncated Normal distribution, with a=0 and b-> inf, 
allowing us to sample from it, and compute the expectation and the variance.

truncnorm: a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
           loc, scale = mu, sigma
           
We get efficient draws using the library rtnorm by C. Lassner, from:
    http://miv.u-strasbg.fr/mazet/rtnorm/
This gives more efficient single draws than scipy.stats.truncnorm.    
    
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
def TN_draw(mu,tau):
    sigma = numpy.float64(1.0) / math.sqrt(tau)
    if tau == 0.:
        return 0.
    d = rtnorm.rtnorm(a=0., b=numpy.inf, mu=mu, sigma=sigma)[0]
    #a,b = -mu/sigma, numpy.inf
    #d = truncnorm(a, b, loc=mu, scale=sigma).rvs(1)[0]
    return d if (d >= 0. and d != numpy.inf and d != -numpy.inf and not numpy.isnan(d)) else 0.
              
# TN expectation        
def TN_expectation(mu,tau):
    sigma = numpy.float64(1.0) / math.sqrt(tau)
    if mu < -30 * sigma:
        exp = 1./(abs(mu)*tau)
    else:
        x = - mu / sigma
        lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
        exp = mu + sigma * lambdax
    return exp if (exp >= 0.0 and exp != numpy.inf and exp != -numpy.inf and not numpy.isnan(exp)) else 0.
       
# TN variance
def TN_variance(mu,tau):
    sigma = numpy.float64(1.0) / math.sqrt(tau)
    if mu < -30 * sigma:
        var = (1./(abs(mu)*tau))**2
    else:
        x = - mu / sigma
        lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
        deltax = lambdax*(lambdax-x)
        var = sigma**2 * ( 1 - deltax )
    return var if (var >= 0.0 and var != numpy.inf and var != -numpy.inf and not numpy.isnan(var)) else 0.       
       
# TN mode
def TN_mode(mu):
    return max(0.0,mu)     
       


'''
# Draw 10000 values and plot. Also plot pdf of Truncated Normal, and regular Normal.
draws = 10000
mu, sigma, tau = 1., 3., 1./9.
lower, upper = (0-mu)/sigma, numpy.inf

time0 = time.time()
# Our implementation
X = [TN_draw(mu,tau) for i in range(0,draws)]
time1 = time.time()
# scipy truncnorm library
X2 = truncnorm(lower, upper, loc=mu, scale=sigma).rvs(draws)
time2 = time.time()
# C. Lassner efficient implementation
X3 = rtnorm.rtnorm(a=0., b=numpy.inf, mu=mu, sigma=sigma, size=draws)
time3 = time.time()
# Draws from normal distribution
N = norm(loc=mu, scale=sigma).rvs(draws)
time4 = time.time()

print "Completed draws. Time taken: ours %s, scipy %s, lassner %s, normal %s." % (time1-time0,time2-time1,time3-time2,time4-time3)

fig, ax = plt.subplots(4, sharex=True)
ax[0].hist(X, bins=range(-10,10), normed=True)
ax[1].hist(X2, bins=range(-10,10), normed=True)
ax[2].hist(X3, bins=range(-10,10), normed=True)
ax[3].hist(N, bins=range(-10,10), normed=True)
plt.show()
'''


'''
mu = -37
std = 1
tau = 1./(std**2)
print TN_expectation(mu,tau), \
      TN_variance(mu,tau), \
      TN_draw(mu,tau)
      
mus = range(-50,-10+1)
exp_exp = [(-1./(mup*tau)) for mup in mus]
exp_tn = [TN_expectation(mup,tau) for mup in mus]
plt.plot(mus,exp_exp,label='Exp')
plt.plot(mus,exp_tn,label='TN')
plt.legend()
'''