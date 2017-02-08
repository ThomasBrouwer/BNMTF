"""
Parallel version of the MatrixCrossValidation class, where we parallelize
the K-fold cross-validation for each parameter.
We now have an extra parameter P for the initialisation, defining the number
of parallel threads we should run.
"""

import mask
from matrix_cross_validation import MatrixCrossValidation

from multiprocessing import Pool
import numpy

attempts_generate_M = 1000


# We try the parameters in parallel. This function either raises an Exception,
# or returns a tuple (parameters,all_performances,average_performances)
def run_fold(params):
    (parameters,X,train,test,method,train_config) = \
        (params['parameters'],params['X'],params['train'],params['test'],params['method'],params['train_config'])    
    performance_dict = run_model(method,X,train,test,parameters,train_config)
    return performance_dict           
    
    
# Method for running the model with the given parameters
def run_model(method,X,train,test,parameters,train_config):
    model = method(X,train,**parameters)
    model.train(**train_config)
    return model.predict(test)


# Class, redefining the run function
class ParallelMatrixCrossValidation(MatrixCrossValidation):
    def __init__(self,method,X,M,K,parameter_search,train_config,file_performance,P):
        MatrixCrossValidation.__init__(self,method,X,M,K,parameter_search,train_config,file_performance)
        self.P = P        
        
    # Run the cross-validation
    def run(self):
        for parameters in self.parameter_search:
            print "Trying parameters %s." % (parameters)
            
            try:
                folds_test = mask.compute_folds_attempts(I=self.I,J=self.J,no_folds=self.K,attempts=attempts_generate_M,M=self.M)
                folds_training = mask.compute_Ms(folds_test)
                
                # We need to put the parameter dict into json to hash it
                self.all_performances[self.JSON(parameters)] = {}
                
                # Create the threads for the folds, and run them
                pool = Pool(self.P)
                all_parameters = [
                    {
                        'parameters' : parameters,
                        'X' : numpy.copy(self.X),
                        'train' : train,
                        'test' : test,
                        'method' : self.method,
                        'train_config' : self.train_config                
                    }
                    for (train,test) in zip(folds_training,folds_test)
                ]
                outputs = pool.map(run_fold,all_parameters)
                pool.close()
                
                for performance_dict in outputs:
                    self.store_performances(performance_dict,parameters)
                    
                self.log(parameters)
                
            except Exception as e:
                self.fout.write("Tried parameters %s but got exception: %s. \n" % (parameters,e))
                
    # Undo the function run_model:
    def run_model(self,train,test,parameters):
        raise Exception("Using wrong method for ParallelMatrixCrossValidation! Use the one defined outside of the class.")