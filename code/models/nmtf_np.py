"""
Non-probabilistic non-negative matrix tri-factorisation, as presented in
"Probabilistic Matrix Tri-Factorisation" (Yoo and Choi, 2009).

We change the notation to match ours: R = FSG.T instead of V = USV.T.

The updates are then:
- Uik <- Uik * (sum_j Vjk * Rij / (Ui dot Vj)) / (sum_j Vjk)
- Vjk <- Vjk * (sum_i Uik * Rij / (Ui dot Vj)) / (sum_i Uik)
Or more efficiently using matrix operations:
- Uik <- Uik * (Mi dot [V.k * Ri / (Ui dot V.T)]) / (Mi dot V.k)
- Vjk <- Vjk * (M.j dot [U.k * R.j / (U dot Vj)]) / (M.j dot U.k)
And realising that elements in each column in U and V are independent:
- U.k <- U.k * sum(M * [V.k * (R / (U dot V.T))], axis=1) / sum(M dot V.k, axis=1)
- V.k <- V.k * sum(M * [U.k * (R / (U dot V.T))], axis=0) / sum(M dot U.k, axis=0)

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of row latent factors
- L, the number of column latent factors
    
Initialisation can be done by running the initialise(init,tauUV) function. We initialise as follows:
- init_S = 'ones'          -> S[i,k] = 1
         = 'random'        -> S[i,k] ~ U(0,1)
         = 'exponential'   -> S[i,k] ~ Exp(expo_prior)
- init_FG = 'ones'          -> F[i,k] = G[j,k] = 1
          = 'random'        -> F[i,k] ~ U(0,1), G[j,l] ~ G(0,1), 
          = 'exponential'   -> F[i,k] ~ Exp(expo_prior), G[j,l] ~ Exp(expo_prior) 
          = 'kmeans'        -> F = KMeans(R,rows)+0.2, G = KMeans(R,columns)+0.2
  where expo_prior is an additional parameter (default 1)
"""

from kmeans.kmeans import KMeans
from distributions.exponential import exponential_draw

import numpy,itertools,math,time

class NMTF:
    def __init__(self,R,M,K,L):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K            
        self.L = L    
        
        self.metrics = ['MSE','R^2','Rp']
                
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
        
        (self.I,self.J) = self.R.shape
        
        self.check_empty_rows_columns() 
        
        # For computing the I-div it is better if unknown values are 1's, not 0's
        self.R_excl_unknown = numpy.empty((self.I,self.J))
        for i,j in itertools.product(range(0,self.I),range(0,self.J)):
            self.R_excl_unknown[i,j] = self.R[i,j] if self.M[i,j] else 1.
                 
                 
    # Raise an exception if an entire row or column is empty
    def check_empty_rows_columns(self):
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j
                 

    """ Initialise F, S and G """    
    def initialise(self,init_S='random',init_FG='random',expo_prior=1.):
        assert init_S in ['ones','random','exponential'], "Unrecognised init option for S: %s." % init_S
        assert init_FG in ['ones','random','exponential','kmeans'], "Unrecognised init option for F,G: %s." % init_FG
        
        if init_S == 'ones':
            self.S = numpy.ones((self.K,self.L))
        elif init_S == 'random':
            self.S = numpy.random.rand(self.K,self.L)
        elif init_S == 'exponential':
            self.S = numpy.empty((self.K,self.L))
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):        
                self.S[k,l] = exponential_draw(expo_prior)
        
        if init_FG == 'ones':
            self.F = numpy.ones((self.I,self.K))
            self.G = numpy.ones((self.J,self.L))
        elif init_FG == 'random':
            self.F = numpy.random.rand(self.I,self.K)
            self.G = numpy.random.rand(self.J,self.L)
        elif init_FG == 'exponential':
            self.F = numpy.empty((self.I,self.K))
            self.G = numpy.empty((self.J,self.L))
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):        
                self.F[i,k] = exponential_draw(expo_prior)
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                self.G[j,l] = exponential_draw(expo_prior)
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
        
        
    """ Update F, S, G for a number of iterations, printing the performances each iteration. """
    def run(self,iterations):
        assert hasattr(self,'F') and hasattr(self,'S') and hasattr(self,'G'), \
            "F, S and G have not been initialised - please run NMTF.initialise() first."        
        
        self.all_times = [] # to plot performance against time
        self.all_performances = {} # for plotting convergence of metrics
        for metric in self.metrics:
            self.all_performances[metric] = []
            
        time_start = time.time()
        for it in range(1,iterations+1):
            # Doing S first gives more interpretable results (F,G ~= [0,1] rather than [0,20])
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                self.update_S(k,l)
                    
            for k in range(0,self.K):
                self.update_F(k)
                
            for l in range(0,self.L):
                self.update_G(l)
               
            self.give_update(it)
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)  
        
        
    """ Method for doing both initialise() and run() """
    def train(self,iterations,init_S='random',init_FG='random',expo_prior=1.):
        self.initialise(init_S=init_S,init_FG=init_FG,expo_prior=expo_prior) 
        self.run(iterations=iterations)
        
                
    """ Updates for F, G, S. """                
    # Compute the dot product of three matrices
    def triple_dot(self,M1,M2,M3):
        return numpy.dot(M1,numpy.dot(M2,M3))
        
    def update_F(self,k):
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        SG = numpy.dot(self.S[k],self.G.T)
        numerator = (self.M * self.R / R_pred * SG).sum(axis=1)
        denominator = (self.M * SG).sum(axis=1)
        self.F[:,k] = self.F[:,k] * numerator / denominator
        
    def update_G(self,l):
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        FS = numpy.dot(self.F,self.S[:,l])
        numerator = ((self.M * self.R / R_pred).T * FS).T.sum(axis=0)
        denominator = (self.M.T * FS).T.sum(axis=0)
        self.G[:,l] = self.G[:,l] * numerator / denominator
        
    def update_S(self,k,l):
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        F_times_G = self.M * numpy.outer(self.F[:,k], self.G[:,l])   
        numerator = (self.R * F_times_G / R_pred).sum()
        denominator = F_times_G.sum()
        self.S[k,l] = self.S[k,l] * numerator / denominator
           
           
    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''
    def predict(self,M_pred):
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}        
        
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
        
    def compute_I_div(self):    
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        return (self.M * ( self.R_excl_unknown * numpy.log( self.R_excl_unknown / R_pred ) - self.R_excl_unknown + R_pred ) ).sum()        
        
        
    """ Give updates and store performances """
    def give_update(self,iteration):    
        perf = self.predict(self.M)
        i_div = self.compute_I_div()
        
        for metric in self.metrics:
            self.all_performances[metric].append(perf[metric])
               
        print "Iteration %s. I-divergence: %s. MSE: %s. R^2: %s. Rp: %s." % (iteration,i_div,perf['MSE'],perf['R^2'],perf['Rp'])
