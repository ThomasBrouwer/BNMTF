"""
Class representing a Truncated Normal distribution, with a=0 and b-> inf, 
allowing us to sample from it, and compute the expectation and the variance.

truncnorm: a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
           loc, scale = mu, sigma
"""
import math, numpy
from scipy.stats import truncnorm, norm

class TruncatedNormal:
    def __init__(self,mu,tau):
        self.mu = float(mu)
        self.tau = float(tau)
        self.sigma = numpy.float64(1.0) / math.sqrt(self.tau)
        
        self.a = - self.mu / self.sigma
        self.b = numpy.inf
        
    # Draw a value for x ~ TruncatedNormal(mu,tau). If we get inf we set it to 0.
    def draw(self):
        d = truncnorm.rvs(a=self.a, b=self.b, loc=self.mu, scale=self.sigma, size=None)
        return d if (d != numpy.inf and not numpy.isnan(d)) else 0.
        
    # Return expectation
    def expectation(self):
        exp = truncnorm.stats(self.a, self.b, loc=self.mu, scale=self.sigma, moments='m')
        return exp if (exp != numpy.inf and not numpy.isnan(exp)) else 0.
        
    # Return variance. The library gives NaN for this due to b->inf, so we compute it ourselves
    def variance(self):
        x = - self.mu / self.sigma
        lambdax = norm.pdf(x)/(1-norm.cdf(x))
        deltax = lambdax*(lambdax-x)
        return self.sigma**2 * ( 1 - deltax )
        
        
'''
# Draw 10000 values and plot. Also plot pdf of Truncated Normal, and regular Normal.
import matplotlib.pyplot as plt
import scipy.stats as stats

mu, sigma, tau = 2., 3., 1./9.
lower, upper = (0-mu)/sigma, numpy.inf
X = [TruncatedNormal(mu,tau).draw() for i in range(0,10000)]
X2 = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)

fig, ax = plt.subplots(3, sharex=True)
ax[0].hist(X, normed=True)
ax[1].hist(X2.rvs(10000), normed=True)
ax[2].hist(N.rvs(10000), normed=True)
plt.show()
'''