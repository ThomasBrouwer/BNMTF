"""
Class for doing model selection for BNMTF, minimising the BIC, AIC, or MSE.
We try an entire grid of K,L values to find the best values.

We expect the following arguments:
- classifier    - a class for BNMTF, with methods: 
                    __init__(R,M,K,L,priors), 
                    initialise(init_S,init_FG), 
                    run(iterations), 
                    quality(metric)         - metric in ['AIC','BIC','loglikelihood','MSE']
                    or quality(metric,burn_in,thinning) for Gibbs
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
- restarts      - we run the classifier this many times and use the one with 
                  the highest log likelihood

The grid search can be started by running search().
If we use Gibbs then we run search(burn_in,thinning).

After that, the values for each metric ('BIC','AIC','loglikelihood','MSE') can 
be obtained using all_values(metric), and the best value of K and L can be 
returned using best_value(metric).

all_values returns a 2D array of size (len(values_K),len(values_L)).

We use the optimised Variational Bayes algorithm for BNMTF.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy

metrics = ['BIC','AIC','loglikelihood','MSE','ELBO']

class GridSearch:
    def __init__(self,classifier,values_K,values_L,R,M,priors,initS,initFG,iterations,restarts=1):
        self.classifier = classifier
        self.values_K = values_K
        self.values_L = values_L
        self.R = R
        self.M = M
        (self.I,self.J) = self.R.shape
        self.priors = priors
        self.initS = initS
        self.initFG = initFG
        self.iterations = iterations
        self.restarts = restarts
        assert self.restarts > 0, "Need at least 1 restart."
        
        self.all_performances = {
            metric : numpy.empty((len(self.values_K),len(self.values_L)))
            for metric in metrics
        }
    
    
    def search(self,burn_in=None,thinning=None):
        for ik,K in enumerate(self.values_K):
            for il,L in enumerate(self.values_L):
                print "Running line search for BNMF. Trying K = %s, L = %s." % (K,L)
                            
                priors = self.priors.copy()
                priors['lambdaF'] = self.priors['lambdaF']*numpy.ones((self.I,K))
                priors['lambdaS'] = self.priors['lambdaS']*numpy.ones((K,L))
                priors['lambdaG'] = self.priors['lambdaG']*numpy.ones((self.J,L))
                
                best_BNMTF = None
                for r in range(0,self.restarts):
                    print "Restart %s for K = %s, L = %s." % (r+1,K,L)    
                    BNMTF = self.classifier(self.R,self.M,K,L,priors)
                    BNMTF.initialise(init_S=self.initS,init_FG=self.initFG)
                    BNMTF.run(iterations=self.iterations)
                    
                    args = {'metric':'loglikelihood'}
                    if burn_in is not None and thinning is not None:
                        args['burn_in'], args['thinning'] = burn_in, thinning
                    
                    if best_BNMTF is None or BNMTF.quality(**args) > best_BNMTF.quality(**args):
                        best_BNMTF = BNMTF
                
                for metric in metrics:
                    if burn_in is not None and thinning is not None:
                        quality = best_BNMTF.quality(metric,burn_in,thinning)
                    else:
                        quality = best_BNMTF.quality(metric)
                    self.all_performances[metric][ik,il] = quality
        
        print "Finished running line search for BNMF."
    
    
    def all_values(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.all_performances[metric]
    
    
    def best_value(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        index, row_length = numpy.argmin(self.all_values(metric)), len(self.values_L)
        return (self.values_K[index / row_length], self.values_L[index % row_length])