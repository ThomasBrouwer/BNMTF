"""
Algorithm for running cross validation with greedy search on a dataset, for the
probabilistic NMTF models (VB, Gibbs, ICM).

Arguments:
- classifier        - the classifier we train
- R                 - the dataset
- M                 - mask matrix
- values_K          - list specifying search range for K
- values_L          - list specifying search range for L
- folds             - number of folds
- priors            - dictionary from hyperparameter names to values
- init_S            - how we initialise S
- init_FG           - how we initialise F and G
- iterations        - number of iterations we run each model
- restarts          - the number of times we try each model when doing model selection
- quality_metric    - the metric we use to measure model quality - MSE, AIC, or BIC
- file_performance  - the file in which we store the performances

We start the search using run(). If we use ICM we use run(minimum_TN=<>)
run(burn_in=<>,thinning=<>).
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

from BNMTF.grid_search.greedy_search_bnmtf import GreedySearch
import numpy, mask

metrics = ['MSE','AIC','BIC'] 

class GreedySearchCrossValidation:
    def __init__(self,classifier,R,M,values_K,values_L,folds,priors,init_S,init_FG,iterations,restarts,quality_metric,file_performance):
        self.classifier = classifier
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M)
        self.values_K = values_K
        self.values_L = values_L
        self.folds = folds
        self.priors = priors
        self.init_FG = init_FG
        self.init_S = init_S
        self.iterations = iterations
        self.restarts = restarts
        self.quality_metric = quality_metric
        
        self.fout = open(file_performance,'w')
        (self.I,self.J) = self.R.shape
        assert (self.R.shape == self.M.shape), "R and M are of different shapes: %s and %s respectively." % (self.R.shape,self.M.shape)
        
        assert self.quality_metric in metrics       
        
        self.performances = {}          # Performances across folds
        
        
    # Run the cross-validation
    def run(self,burn_in=None,thinning=None,minimum_TN=None):
        folds_test = mask.compute_folds(self.I,self.J,self.folds,self.M)
        folds_training = mask.compute_Ms(folds_test)

        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            print "Fold %s." % (i+1)
            
            # Run the greedy grid search
            greedy_search = GreedySearch(
                classifier=self.classifier,
                values_K=self.values_K,
                values_L=self.values_L,
                R=self.R,
                M=self.M,
                priors=self.priors,
                initS=self.init_S,
                initFG=self.init_FG,
                iterations=self.iterations,
                restarts=self.restarts)
            greedy_search.search(self.quality_metric,burn_in=burn_in,thinning=thinning,minimum_TN=minimum_TN)
            
            # Store the model fits, and find the best one according to the metric    
            all_performances = greedy_search.all_values(metric=self.quality_metric)
            self.fout.write("All model fits for fold %s, metric %s: %s.\n" % (i+1,self.quality_metric,all_performances)) 
            self.fout.flush()
            
            best_KL = greedy_search.best_value(metric=self.quality_metric)
            self.fout.write("Best K,L for fold %s: %s.\n" % (i+1,best_KL))
            
            # Train a model with this K and measure performance on the test set
            performance = self.run_model(train,test,best_KL[0],best_KL[1],burn_in=burn_in,thinning=thinning,minimum_TN=minimum_TN)
            self.fout.write("Performance: %s.\n\n" % performance)
            self.fout.flush()
            
            
    # Initialises and runs the model, and returns the performance on the test set
    def run_model(self,train,test,K,L,burn_in=None,thinning=None,minimum_TN=None):
        model = self.classifier(
            R=self.R,
            M=train,
            K=K,
            L=L,
            priors=self.priors
        )
        model.initialise(self.init_S,self.init_FG)
        
        if minimum_TN is None:
            model.run(self.iterations)
        else:
            model.run(self.iterations,minimum_TN=minimum_TN)
            
        if burn_in is None or thinning is None:
            return model.predict(test)
        else:
            return model.predict(test,burn_in,thinning)