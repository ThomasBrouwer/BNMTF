"""
Variational Bayesian inference for non-negative matrix factorisation.
We optimise the updates s.t. we compute each column of U and V using matrix
operations, rather than each element individually.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of latent factors
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaU' = [[lambdaUik]], 'lambdaV' = [[lambdaVjk]] },
    a dictionary defining the priors over tau, U, V.
    
Initialisation can be done by running the initialise(init,tauUV) function. We initialise as follows:
- init = 'exp'       -> muU[i,k] = 1/lambdaU[i,k], muV[j,k] = 1/lambdaV[j,k]
       = 'random'    -> muU[i,k] ~ Exp(lambdaU[i,k]), muV[j,k] ~ Exp(lambdaV[j,k]), 
- tauU[i,k] = tauV[j,k] = 1 if tauUV = {}, else tauU = tauUV['tauU'], tauV = tauUV['tauV']
- alpha_s, beta_s using updates of model

Usage of class:
    BNMF = bnmf_vb(R,M,K,priors)
    BNMF.initisalise(init)      
    BNMF.run(iterations)
Or:
    BNMF = bnmf_vb(R,M,K,priors)
    BNMF.train(init,iterations)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
The performances of all iterations are stored in BNMF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'        -> return Bayesian Information Criterion
         = 'AIC'        -> return Afaike Information Criterion
         = 'MSE'        -> return Mean Square Error
         = 'ELBO'       -> return Evidence Lower Bound
(we want to maximise these values)
"""

from distributions.gamma import gamma_expectation, gamma_expectation_log, gamma_draw
from distributions.truncated_normal_vector import TN_vector_expectation, TN_vector_variance
from distributions.exponential import exponential_draw

import numpy, itertools, math, scipy, time
from scipy.stats import norm
import matplotlib.pyplot as plt

class bnmf_vb_optimised:
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
        
        # If lambdaU or lambdaV are an integer rather than a numpy array, we make it into one using that value
        if self.lambdaU.shape == ():
            self.lambdaU = self.lambdaU * numpy.ones((self.I,self.K))
        if self.lambdaV.shape == ():
            self.lambdaV = self.lambdaV * numpy.ones((self.J,self.K))
        
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


    # Initialise U, V, and tau. 
    def initialise(self,init='exp',tauUV={}):
        self.tauU = tauUV['tauU'] if 'tauU' in tauUV else numpy.ones((self.I,self.K))
        self.tauV = tauUV['tauV'] if 'tauV' in tauUV else numpy.ones((self.J,self.K))
        
        assert init in ['exp','random'], "Unrecognised init option for F,G: %s." % init
        self.muU, self.muV = 1./self.lambdaU, 1./self.lambdaV
        if init == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):        
                self.muU[i,k] = exponential_draw(self.lambdaU[i,k])
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.muV[j,k] = exponential_draw(self.lambdaV[j,k])  
        
        # Initialise the expectations and variances
        self.expU, self.varU = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.expV, self.varV = numpy.zeros((self.J,self.K)), numpy.zeros((self.J,self.K))
        
        for k in xrange(0,self.K):
            self.update_exp_U(k)
        for k in xrange(0,self.K):
            self.update_exp_V(k)
        
        # Update tau using the updates
        self.update_tau()
        #self.alpha_s, self.beta_s = self.alpha, self.beta
        self.update_exp_tau()
        

    # Run the Gibbs sampler
    def run(self,iterations):
        self.all_exp_tau = []  # to check for convergence 
        self.all_times = [] # to plot performance against time
        
        metrics = ['MSE','R^2','Rp']
        self.all_performances = {} # for plotting convergence of metrics
        for metric in metrics:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(0,iterations):
            for k in xrange(0,self.K):
                self.update_U(k)
                self.update_exp_U(k)    
                
            for k in xrange(0,self.K):
                self.update_V(k)
                self.update_exp_V(k)
                
            self.update_tau()
            self.update_exp_tau()
            self.all_exp_tau.append(self.exptau)
            
            perf, elbo = self.predict(self.M), self.elbo()
            for metric in metrics:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,elbo,perf['MSE'],perf['R^2'],perf['Rp'])
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
            
        return
        
        
    # Method for doing both initialise() and run() 
    def train(self,iterations,init_UV='random'):
        self.initialise(init_UV=init_UV) 
        self.run(iterations=iterations)    
        
        
    # Compute the ELBO
    def elbo(self):
        return self.size_Omega / 2. * ( self.explogtau - math.log(2*math.pi) ) \
             - self.exptau / 2. * self.exp_square_diff() \
             + numpy.log(self.lambdaU).sum() - ( self.lambdaU * self.expU ).sum() \
             + numpy.log(self.lambdaV).sum() - ( self.lambdaV * self.expV ).sum() \
             + self.alpha * math.log(self.beta) - scipy.special.gammaln(self.alpha) \
             + (self.alpha - 1.)*self.explogtau - self.beta * self.exptau \
             - self.alpha_s * math.log(self.beta_s) + scipy.special.gammaln(self.alpha_s) \
             - (self.alpha_s - 1.)*self.explogtau + self.beta_s * self.exptau \
             - .5*numpy.log(self.tauU).sum() + self.I*self.K/2.*math.log(2*math.pi) \
             + numpy.log(0.5*scipy.special.erfc(-self.muU*numpy.sqrt(self.tauU)/math.sqrt(2))).sum() \
             + ( self.tauU / 2. * ( self.varU + (self.expU - self.muU)**2 ) ).sum() \
             - .5*numpy.log(self.tauV).sum() + self.J*self.K/2.*math.log(2*math.pi) \
             + numpy.log(0.5*scipy.special.erfc(-self.muV*numpy.sqrt(self.tauV)/math.sqrt(2))).sum() \
             + ( self.tauV / 2. * ( self.varV + (self.expV - self.muV)**2 ) ).sum()
        
        
    # Update the parameters for the distributions
    def update_tau(self):   
        self.alpha_s = self.alpha + self.size_Omega/2.0
        self.beta_s = self.beta + 0.5*self.exp_square_diff()
        
    def exp_square_diff(self): # Compute: sum_Omega E_q(U,V) [ ( Rij - Ui Vj )^2 ]
        return(self.M *( ( self.R - numpy.dot(self.expU,self.expV.T) )**2 + \
                         ( numpy.dot(self.varU+self.expU**2, (self.varV+self.expV**2).T) - numpy.dot(self.expU**2,(self.expV**2).T) ) ) ).sum()
        
    def update_U(self,k):       
        self.tauU[:,k] = self.exptau*(self.M*( self.varV[:,k] + self.expV[:,k]**2 )).sum(axis=1) #sum over j, so rows
        self.muU[:,k] = 1./self.tauU[:,k] * (-self.lambdaU[:,k] + self.exptau*(self.M * ( (self.R-numpy.dot(self.expU,self.expV.T)+numpy.outer(self.expU[:,k],self.expV[:,k]))*self.expV[:,k] )).sum(axis=1)) 
        
    def update_V(self,k):
        self.tauV[:,k] = self.exptau*(self.M.T*( self.varU[:,k] + self.expU[:,k]**2 )).T.sum(axis=0) #sum over i, so columns
        self.muV[:,k] = 1./self.tauV[:,k] * (-self.lambdaV[:,k] + self.exptau*(self.M.T * ( (self.R-numpy.dot(self.expU,self.expV.T)+numpy.outer(self.expU[:,k],self.expV[:,k])).T*self.expU[:,k] )).T.sum(axis=0)) 
        
        
    # Update the expectations and variances
    def update_exp_U(self,k):
        #tn = TruncatedNormalVector(self.muU[:,k],self.tauU[:,k])
        #self.expU[:,k] = tn.expectation()
        #self.varU[:,k] = tn.variance()
        self.expU[:,k] = TN_vector_expectation(self.muU[:,k],self.tauU[:,k])
        self.varU[:,k] = TN_vector_variance(self.muU[:,k],self.tauU[:,k])
        
    def update_exp_V(self,k):
        #tn = TruncatedNormalVector(self.muV[:,k],self.tauV[:,k])
        #self.expV[:,k] = tn.expectation()
        #self.varV[:,k] = tn.variance()
        self.expV[:,k] = TN_vector_expectation(self.muV[:,k],self.tauV[:,k])
        self.varV[:,k] = TN_vector_variance(self.muV[:,k],self.tauV[:,k])
        
    def update_exp_tau(self):
        self.exptau = gamma_expectation(self.alpha_s,self.beta_s)
        self.explogtau = gamma_expectation_log(self.alpha_s,self.beta_s)


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred):
        R_pred = numpy.dot(self.expU,self.expV.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
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
        
        
    # Functions for model selection, measuring the goodness of fit vs model complexity
    def quality(self,metric):
        assert metric in ['loglikelihood','BIC','AIC','MSE','ELBO'], 'Unrecognised metric for model quality: %s.' % metric
        log_likelihood = self.log_likelihood()
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + (self.I*self.K+self.J*self.K) * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * (self.I*self.K+self.J*self.K)
        elif metric == 'MSE':
            R_pred = numpy.dot(self.expU,self.expV.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return self.elbo()
        
    def log_likelihood(self):
        # Return the likelihood of the data given the trained model's parameters
        return self.size_Omega / 2. * ( self.explogtau - math.log(2*math.pi) ) \
             - self.exptau / 2. * (self.M*( self.R - numpy.dot(self.expU,self.expV.T))**2).sum()