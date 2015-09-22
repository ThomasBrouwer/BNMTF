"""
Gibbs sampler for non-negative matrix factorisation, as introduced by Schmidt
et al. (2009).
Optimised to draw all columns in parallel.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of latent factors
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaU' = [[lambdaUik]], 'lambdaV' = [[lambdaVjk]] },
    a dictionary defining the priors over tau, U, V.
    
Initialisation can be done by running the initialise() function, with argument init:
- init='random' -> draw initial values randomly from priors Exp, Gamma
- init='exp'    -> use the expectation of the priors Exp, Gamma
Alternatively, you can define your own initial values for U, V, and tau.

Usage of class:
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initisalise(init)
    BNMF.run(iterations)
Or:
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.train(init,iterations)
    
This returns a tuple (Us,Vs,taus) of lists of U, V, tau values - of size <iterations>.

The expectation can be computed by specifying a burn-in and thinning rate, and using:
    BNMF.approx_expectation(burn_in,thinning)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred,burn_in,thinning)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
"""

from distributions.exponential import Exponential
from distributions.gamma import Gamma
from distributions.truncated_normal_vector import TruncatedNormalVector

import numpy, itertools, math, time

class bnmf_gibbs_optimised:
    def __init__(self,R,M,K,priors):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alpha, self.beta, self.lambdaU, self.lambdaV = \
            float(priors['alpha']), float(priors['beta']), numpy.array(priors['lambdaU']), numpy.array(priors['lambdaV'])
        
        assert self.lambdaU.shape == (self.I,self.K), "Prior matrix lambdaU has the wrong shape: %s instead of (%s, %s)." % (self.lambdaU.shape,self.I,self.K)
        assert self.lambdaV.shape == (self.J,self.K), "Prior matrix lambdaV has the wrong shape: %s instead of (%s, %s)." % (self.lambdaV.shape,self.J,self.K)
            
        
    # Raise an exception if an entire row or column is empty
    def check_empty_rows_columns(self):
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j


    # Initialise and run the sampler
    def train(self,init,iterations):
        self.initialise(init=init)
        return self.run(iterations)


    # Initialise U, V, and tau. If init='random', draw values from an Exp and Gamma distribution. If init='exp', set it to the expectation values.
    def initialise(self,init='random'):
        assert init in ['random','exp'], "Unknown initialisation option: %s. Should be 'random' or 'exp'." % init
        self.U = numpy.zeros((self.I,self.K))
        self.V = numpy.zeros((self.J,self.K))
        
        if init == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.U[i,k] = Exponential(self.lambdaU[i][k]).draw()
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = Exponential(self.lambdaV[j][k]).draw()
            self.tau = Gamma(self.alpha,self.beta).draw()
            
        elif init == 'exp':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.U[i,k] = 1.0/self.lambdaU[i][k]
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = 1.0/self.lambdaV[j][k]
            self.tau = self.alpha/self.beta


    # Run the Gibbs sampler
    def run(self,iterations):
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))   
        self.all_tau = numpy.zeros(iterations)
        
        for it in range(0,iterations):      
            for k in range(0,self.K):   
                tauUk = self.tauU(k)
                muUk = self.muU(tauUk,k)
                self.U[:,k] = TruncatedNormalVector(muUk,tauUk).draw()
                
            for k in range(0,self.K):
                tauVk = self.tauV(k)
                muVk = self.muV(tauVk,k)
                self.V[:,k] = TruncatedNormalVector(muVk,tauVk).draw()
                
            self.tau = Gamma(self.alpha_s(),self.beta_s()).draw()
            
            self.all_U[it], self.all_V[it], self.all_tau[it] = numpy.copy(self.U), numpy.copy(self.V), self.tau
            perf = self.predict_while_running()
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
            
        return (self.all_U, self.all_V, self.all_tau)
        
        
    # Compute the parameters for the distributions we sample from
    def alpha_s(self):   
        return self.alpha + self.size_Omega/2.0
    
    def beta_s(self):   
        return self.beta + 0.5*(self.M*(self.R-numpy.dot(self.U,self.V.T))**2).sum()
        
    def tauU(self,k):       
        return self.tau*(self.M*self.V[:,k]**2).sum(axis=1)
        
    def muU(self,tauUk,k):
        return 1./tauUk * (-self.lambdaU[:,k] + self.tau*(self.M * ( (self.R-numpy.dot(self.U,self.V.T)+numpy.outer(self.U[:,k],self.V[:,k]))*self.V[:,k] )).sum(axis=1)) 
        
    def tauV(self,k):
        return self.tau*(self.M.T*self.U[:,k]**2).T.sum(axis=0)
        
    def muV(self,tauVk,k):
        return 1./tauVk * (-self.lambdaV[:,k] + self.tau*(self.M.T * ( (self.R-numpy.dot(self.U,self.V.T)+numpy.outer(self.U[:,k],self.V[:,k])).T*self.U[:,k] )).T.sum(axis=0)) 


    # Return the average value for U, V, tau - i.e. our approximation to the expectations. 
    # Throw away the first <burn_in> samples, and then use every <thinning>th after.
    def approx_expectation(self,burn_in,thinning):
        indices = range(burn_in,len(self.all_U),thinning)
        exp_U = numpy.array([self.all_U[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_V = numpy.array([self.all_V[i] for i in indices]).sum(axis=0) / float(len(indices))  
        exp_tau = sum([self.all_tau[i] for i in indices]) / float(len(indices))
        return (exp_U, exp_V, exp_tau)


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred,burn_in,thinning):
        (exp_U,exp_V,_) = self.approx_expectation(burn_in,thinning)
        R_pred = numpy.dot(exp_U,exp_V.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
    def predict_while_running(self):
        R_pred = numpy.dot(self.U,self.V.T)
        MSE = self.compute_MSE(self.M,self.R,R_pred)
        R2 = self.compute_R2(self.M,self.R,R_pred)    
        Rp = self.compute_Rp(self.M,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    # Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation)
    def compute_MSE(self,M,R,R_pred):
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf
        
    def compute_Rp(self,M,R,R_pred):
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))