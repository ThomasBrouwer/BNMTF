# Nested cross-validation for the KBMF method.

source("cross_val_kbmf.R")

kbmf_nested_cross_validation <- function(Kx, Kz, Y, R_values, K) {
	# Split the dataset up into K folds
	sets = create_train_test_sets(split_dataset(Y, K), Y)
	training_sets = sets[[1]]
	test_sets = sets[[2]]

	MSEs = list()
	R2s = list()
	Rps = list()
	for (f in seq(training_sets)) {
		print(sprintf("FOLD %i. Now running cross-validation to find best R.", f))
		train = training_sets[[f]]
		test = test_sets[[f]]

		# Run X-val on each training set
		results = kbmf_cross_validation(Kx, Kz, train, R_values, K)

		# Use the best value for R to train and evaluate on the test set
		best_R = results[[1]]
		state <- kbmf_regression_train(Kx, Kz, train, best_R)
		prediction <- kbmf_regression_test(Kx, Kz, state)$Y$mu
		
		MSE = mean((prediction - test)^2, na.rm=TRUE )
		mean_test = mean( test, na.rm=TRUE )
		R2 = 1 - ( sum( (test - prediction)^2, na.rm=TRUE ) / sum( (test - mean_test)^2, na.rm=TRUE ) )
		mean_pred = mean( prediction, na.rm=TRUE )
		Rp = cor(c(test),c(prediction),use='pairwise.complete.obs',method='pearson')
		#Rp = sum( (test - mean_test) * (prediction - mean_pred) , na.rm=TRUE ) / ( sqrt( sum( (test - mean_test)^2 , na.rm=TRUE ) ) * sqrt( sum( (prediction - mean_pred)^2 , na.rm=TRUE ) ) )
		print(sprintf("Performance on fold %i: MSE=%.4f, R^2=%.4f, Rp=%.4f.", f,MSE,R2,Rp))

		# Store the performance
		MSEs = c(MSEs,MSE)
		R2s = c(R2s,R2)
		Rps = c(Rps,Rp)
	}

	# Print all performances
	print(sprintf("All performances nested cross-validation: MSE=%.4f, R^2=%.4f, Rp=%.4f.",unlist(MSEs),unlist(R2s),unlist(Rps)))
	
	# Compute the average performances, and return that.
	average_MSE = mean(unlist(MSEs))
	average_R2 = mean(unlist(R2s))
	average_Rp = mean(unlist(Rps))
	print(sprintf("Performances nested cross-validation: MSE=%.4f, R^2=%.4f, Rp=%.4f.",average_MSE,average_R2,average_Rp))
	return(list(average_MSE, average_R2, average_Rp))
}


