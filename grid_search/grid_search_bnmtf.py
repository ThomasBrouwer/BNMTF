"""
Class for doing model selection for BNMTF, maximising the BIC, AIC, or log likelihood.

We expect the following arguments:
- values_K      - a list of values for K
- values_L      - a list of values for L
- R             - the dataset
- M             - the mask matrix
- prior         - the prior values for BNMF. This should be a dictionary of the form:
                    { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
                  where lambdaF, lambdaS, and lambdaG are a single value.
- initFG        - the initialisation of F and G - 'kmeans', 'exp' or 'random'
- initS        - the initialisation of S - 'exp' or 'random'
- iterations    - number of iterations to run 

The line search can be started by running search().

After that, the values for each metric ('BIC','AIC','loglikelihood') can be
obtained using all_values(metric), and the best value of K and L can be 
returned using best_value(metric).

all_values returns a 2D array of size (len(values_K),len(values_L)).

We use the optimised Variational Bayes algorithm for BNMTF.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
import numpy

metrics = ['BIC','AIC','loglikelihood']

class GridSearch:
    def __init__(self,values_K,values_L,R,M,priors,initS,initFG,iterations):
        self.values_K = values_K
        self.values_L = values_L
        self.R = R
        self.M = M
        (self.I,self.J) = self.R.shape
        self.priors = priors
        self.initS = initS
        self.initFG = initFG
        self.iterations = iterations
        
        self.all_performances = {
            metric : numpy.empty((len(self.values_K),len(self.values_L)))
            for metric in metrics
        }
    
    
    def search(self):
        for ik,K in enumerate(self.values_K):
            for il,L in enumerate(self.values_L):
                print "Running line search for BNMF. Trying K = %s, L = %s." % (K,L)
                            
                priors = self.priors.copy()
                priors['lambdaF'] = self.priors['lambdaF']*numpy.ones((self.I,K))
                priors['lambdaS'] = self.priors['lambdaS']*numpy.ones((K,L))
                priors['lambdaG'] = self.priors['lambdaG']*numpy.ones((self.J,L))
                
                BNMF = bnmtf_vb_optimised(self.R,self.M,K,L,priors)
                BNMF.initialise(init_S=self.initS,init_FG=self.initFG)
                BNMF.run(iterations=self.iterations)
                
                for metric in metrics:
                    quality = BNMF.quality(metric)
                    self.all_performances[metric][ik,il] = quality
        
        print "Finished running line search for BNMF."
    
    
    def all_values(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.all_performances[metric]
    
    
    def best_value(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        index, row_length = numpy.argmax(self.all_values(metric)), len(self.values_L)
        return (self.values_K[index / row_length], self.values_L[index % row_length])