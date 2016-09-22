"""
Iterated Conditional Modes for MAP non-negative matrix tri-factorisation.

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
    NMF = nmf_icm(R,M,K,priors)
    NMF.initisalise(init)
    NMF.run(iterations)
Or:
    NMF = nmf_icm(R,M,K,priors)
    NMF.train(init,iterations)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = NMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
The performances of all iterations are stored in NMF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'        -> return Bayesian Information Criterion
         = 'AIC'        -> return Afaike Information Criterion
         = 'MSE'        -> return Mean Square Error
(we want to maximise these values)
"""

from distributions.exponential import exponential_draw
from distributions.gamma import gamma_mode
from distributions.truncated_normal_vector import TN_vector_mode

import numpy, itertools, math, time

class nmf_icm:
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
                self.U[i,k] = exponential_draw(self.lambdaU[i][k])
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = exponential_draw(self.lambdaV[j][k])
            
        elif init == 'exp':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.U[i,k] = 1.0/self.lambdaU[i][k]
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = 1.0/self.lambdaV[j][k]
        
        self.tau = gamma_mode(self.alpha_s(), self.beta_s())
       

    # Run the Gibbs sampler
    def run(self,iterations,minimum_TN=0.):   
        self.all_tau = numpy.zeros(iterations) # to plot convergence
        self.all_times = [] # to plot performance against time
        
        metrics = ['MSE','R^2','Rp']
        self.all_performances = {} # for plotting convergence of metrics
        for metric in metrics:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(0,iterations):      
            for k in range(0,self.K):   
                tauUk = self.tauU(k)
                muUk = self.muU(tauUk,k)
                self.U[:,k] = TN_vector_mode(muUk) 
                self.U[:,k] = numpy.maximum(self.U[:,k],minimum_TN*numpy.ones(self.I))
                
            for k in range(0,self.K):
                tauVk = self.tauV(k)
                muVk = self.muV(tauVk,k)
                self.V[:,k] = TN_vector_mode(muVk) 
                self.V[:,k] = numpy.maximum(self.V[:,k],minimum_TN*numpy.ones(self.J))
                
            self.tau = gamma_mode(self.alpha_s(),self.beta_s())
            self.all_tau[it] = self.tau
            
            perf = self.predict(self.M)
            for metric in metrics:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
            
        return
        
        
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


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred):
        R_pred = numpy.dot(self.U,self.V.T)
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
            R_pred = numpy.dot(self.U,self.V.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return 0.
        
    def log_likelihood(self):
        # Return the likelihood of the data given the trained model's parameters
        return self.size_Omega / 2. * ( math.log(self.tau) - math.log(2*math.pi) ) \
             - self.tau / 2. * (self.M*( self.R - numpy.dot(self.U,self.V.T))**2).sum()