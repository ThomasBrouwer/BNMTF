"""
Iterated Conditional Modes for MAP non-negative matrix tri-factorisation.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of row clusters
- L, the number of column clusters
- priors = { 'alpha' = alpha_R, 'beta' = beta_R, 'lambdaF' = [[lambdaFik]], 'lambdaS' = [[lambdaSkl]], 'lambdaG' = [[lambdaGjl]] },
    a dictionary defining the priors over tau, F, S, G.
    
Initialisation can be done by running the initialise() function, with argument init:
- init='random' -> draw initial values randomly from priors Exp, Gamma
- init='exp'    -> use the expectation of the priors Exp, Gamma
Alternatively, you can define your own initial values for F, S, G, and tau.

Usage of class:
    BNMF = bnmf_gibbs(R,M,K,L,priors)
    BNMF.initisalise(init)
    BNMF.run(iterations)
Or:
    BNMF = bnmf_gibbs(R,M,K,L,priors)
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
(we want to maximise these values)
"""

from kmeans.kmeans import KMeans
from distributions.exponential import exponential_draw
from distributions.gamma import gamma_mode
from distributions.truncated_normal import TN_mode
from distributions.truncated_normal_vector import TN_vector_mode

import numpy, itertools, math, time

class nmtf_icm:
    def __init__(self,R,M,K,L,priors):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        self.L = L
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alpha, self.beta, self.lambdaF, self.lambdaS, self.lambdaG = \
            float(priors['alpha']), float(priors['beta']), numpy.array(priors['lambdaF']), numpy.array(priors['lambdaS']), numpy.array(priors['lambdaG'])
        
        # If lambdaF, lambdaS, or lambdaG are an integer rather than a numpy array, we make it into one using that value
        if self.lambdaF.shape == ():
            self.lambdaF = self.lambdaF * numpy.ones((self.I,self.K))
        if self.lambdaS.shape == ():
            self.lambdaS = self.lambdaS * numpy.ones((self.K,self.L))
        if self.lambdaG.shape == ():
            self.lambdaG = self.lambdaG * numpy.ones((self.J,self.L))
        
        assert self.lambdaF.shape == (self.I,self.K), "Prior matrix lambdaF has the wrong shape: %s instead of (%s, %s)." % (self.lambdaF.shape,self.I,self.K)
        assert self.lambdaS.shape == (self.K,self.L), "Prior matrix lambdaS has the wrong shape: %s instead of (%s, %s)." % (self.lambdaS.shape,self.K,self.L)
        assert self.lambdaG.shape == (self.J,self.L), "Prior matrix lambdaG has the wrong shape: %s instead of (%s, %s)." % (self.lambdaG.shape,self.J,self.L)
            
        
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
    def initialise(self,init_S='random',init_FG='random'):
        assert init_S in ['random','exp'], "Unknown initialisation option for S: %s. Should be 'random' or 'exp'." % init_S
        assert init_FG in ['random','exp','kmeans'], "Unknown initialisation option for S: %s. Should be 'random', 'exp', or 'kmeans." % init_FG
        
        self.S = 1./self.lambdaS
        if init_S == 'random':
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):  
                self.S[k,l] = exponential_draw(self.lambdaS[k,l])
                
        self.F, self.G = 1./self.lambdaF, 1./self.lambdaG
        if init_FG == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):        
                self.F[i,k] = exponential_draw(self.lambdaF[i,k])
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                self.G[j,l] = exponential_draw(self.lambdaG[j,l])
        elif init_FG == 'kmeans':
            print "Initialising F using KMeans."
            kmeans_F = KMeans(self.R,self.M,self.K)
            kmeans_F.initialise()
            kmeans_F.cluster()
            self.F = kmeans_F.clustering_results + 0.2            
            
            print "Initialising G using KMeans."
            kmeans_G = KMeans(self.R.T,self.M.T,self.L)   
            kmeans_G.initialise()
            kmeans_G.cluster()
            self.G = kmeans_G.clustering_results + 0.2

        self.tau = gamma_mode(self.alpha_s(), self.beta_s())


    # Run the Gibbs sampler
    def run(self,iterations,minimum_TN=0.):  
        self.all_tau = numpy.zeros(iterations)
        self.all_times = [] # to plot performance against time
        
        metrics = ['MSE','R^2','Rp']
        self.all_performances = {} # for plotting convergence of metrics
        for metric in metrics:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(0,iterations):            
            for k in range(0,self.K):
                tauFk = self.tauF(k)
                muFk = self.muF(tauFk,k)
                self.F[:,k] = TN_vector_mode(muFk)
                self.F[:,k] = numpy.maximum(self.F[:,k],minimum_TN*numpy.ones(self.I))
                
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                tauSkl = self.tauS(k,l)
                muSkl = self.muS(tauSkl,k,l)
                self.S[k,l] = TN_mode(muSkl)
                self.S[k,l] = max(self.S[k,l],minimum_TN)
                
            for l in range(0,self.L):
                tauGl = self.tauG(l)
                muGl = self.muG(tauGl,l)
                self.G[:,l] = TN_vector_mode(muGl)
                self.G[:,l] = numpy.maximum(self.G[:,l],minimum_TN*numpy.ones(self.J))
                
            self.tau = gamma_mode(self.alpha_s(),self.beta_s())
            self.all_tau[it] = self.tau
            
            perf = self.predict(self.M)
            for metric in metrics:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
        
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
            
        return 
        

    # Compute the dot product of three matrices
    def triple_dot(self,M1,M2,M3):
        return numpy.dot(M1,numpy.dot(M2,M3))
        
        
    # Compute the parameters for the distributions we sample from
    def alpha_s(self):   
        return self.alpha + self.size_Omega/2.0
    
    def beta_s(self):   
        return self.beta + 0.5*(self.M*(self.R-self.triple_dot(self.F,self.S,self.G.T))**2).sum()
        
    def tauF(self,k):       
        return self.tau * ( self.M * numpy.dot(self.S[k],self.G.T)**2 ).sum(axis=1)
        
    def muF(self,tauFk,k):
        return 1./tauFk * (-self.lambdaF[:,k] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+numpy.outer(self.F[:,k],numpy.dot(self.S[k],self.G.T)))*numpy.dot(self.S[k],self.G.T) )).sum(axis=1)) 
        
    def tauS(self,k,l):       
        return self.tau * ( self.M * numpy.outer(self.F[:,k]**2,self.G[:,l]**2) ).sum()
        
    def muS(self,tauSkl,k,l):
        return 1./tauSkl * (-self.lambdaS[k,l] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+self.S[k,l]*numpy.outer(self.F[:,k],self.G[:,l]))*numpy.outer(self.F[:,k],self.G[:,l]) )).sum()) 
        
    def tauG(self,l):       
        return self.tau * ( self.M.T * numpy.dot(self.F,self.S[:,l])**2 ).T.sum(axis=0)
        
    def muG(self,tauGl,l):
        return 1./tauGl * (-self.lambdaG[:,l] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+numpy.outer(numpy.dot(self.F,self.S[:,l]),self.G[:,l])).T * numpy.dot(self.F,self.S[:,l]) ).T).sum(axis=0)) 
        

    # Return the average value for U, V, tau - i.e. our approximation to the expectations. 
    # Throw away the first <burn_in> samples, and then use every <thinning>th after.
    def approx_expectation(self,burn_in,thinning):
        indices = range(burn_in,len(self.all_F),thinning)
        exp_F = numpy.array([self.all_F[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_S = numpy.array([self.all_S[i] for i in indices]).sum(axis=0) / float(len(indices))   
        exp_G = numpy.array([self.all_G[i] for i in indices]).sum(axis=0) / float(len(indices))  
        exp_tau = sum([self.all_tau[i] for i in indices]) / float(len(indices))
        return (exp_F, exp_S, exp_G, exp_tau)


    # Compute the expectation of U and V, and use it to predict missing values
    def predict(self,M_pred):
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
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
            return - 2 * log_likelihood + (self.I*self.K+self.K*self.L+self.J*self.L) * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * (self.I*self.K+self.K*self.L+self.J*self.L)
        elif metric == 'MSE':
            R_pred = self.triple_dot(self.F,self.S,self.G.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return 0.
        
    def log_likelihood(self):
        # Return the likelihood of the data given the trained model's parameters
        return self.size_Omega / 2. * ( math.log(self.tau) - math.log(2*math.pi) ) \
             - self.tau / 2. * (self.M*( self.R - self.triple_dot(self.F,self.S,self.G.T) )**2).sum()