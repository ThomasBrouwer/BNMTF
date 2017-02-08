# Try different values for the latent dimension R, using K-fold cross-validation, and return the performances of the predictions for different values of R.

setwd("../kbmf/")
source("kbmf_regression_train.R")
source("kbmf_regression_test.R")

split_dataset <- function(Y, K) {
	nrows <- nrow(Y)
	ncols <- ncol(Y)
	entries <- which(!is.na(Y)) # gives indices of !NA entries

	# Split dataset
	random_entries <- sample(entries, replace = FALSE)
	split_entries <- split(random_entries, 1:K) 

	# Create folds and add to list
	folds <- list()
	for (i in seq(split_entries)) {
		# For each index, set that entry to the value in entries
		new_fold = matrix(data=NA, nrow=nrows, ncol=ncols)
		entries_fold = split_entries[[i]]
		for (j in seq(entries_fold)) {
			index = entries_fold[j]
			new_fold[index] = Y[index]
		}
		folds[[i]] = new_fold
	}
	return(folds)
}

create_train_test_sets <- function(folds,Y) {
	nrows <- nrow(Y)
	ncols <- ncol(Y)

	# Use the folds to create the training and test sets
	K <- length(folds)
	train_sets <- list()
	test_sets <- list()

	for (i in seq(folds)) {
		# Test set is this fold, training is the other four - so set these entries to NA
		fold = folds[[i]]
		test_sets[[i]] = fold

		train_sets[[i]] = matrix(data=Y, nrow=nrows, ncol=ncols) #copy over Y, and set values to NA
		entries <- which(!is.na(fold))
		for (j in seq(entries)) {
			index = entries[j]
			train_sets[[i]][index] = NA
		}
	}

	return(list(train_sets,test_sets))
}

kbmf_cross_validation <- function(Kx, Kz, Y, R_values, K) {
	# For each value R in R_values, split the data into K folds, and use K-1 folds to train and test on the remaining folds
	all_MSEs = list()
	all_R2s = list()
	all_Rps = list()
	for (i in seq(R_values)) {
		R = R_values[[i]]
    		print(sprintf("Trying R=%i", R))

		sets = create_train_test_sets(split_dataset(Y, K), Y)
		training_sets = sets[[1]]
		test_sets = sets[[2]]

		# For each fold, test the performance
		MSEs = list()
		R2s = list()
		Rps = list()
		for (f in seq(training_sets)) {
			print(sprintf("Fold %i.", f))

			train = training_sets[[f]]
			test = test_sets[[f]]

			state <- kbmf_regression_train(Kx, Kz, train, R)
			prediction <- kbmf_regression_test(Kx, Kz, state)$Y$mu

			MSE = mean((prediction - test)^2, na.rm=TRUE )
			mean_test = mean(test, na.rm=TRUE )
			R2 = 1 - ( sum( (test - prediction)^2, na.rm=TRUE ) / sum( (test - mean_test)^2, na.rm=TRUE ) )
			mean_pred = mean( prediction, na.rm=TRUE )
			Rp = cor(c(test),c(prediction),use='pairwise.complete.obs',method='pearson')
			#Rp = sum( (test - mean_test) * (prediction - mean_pred) , na.rm=TRUE ) / ( sqrt( sum( (test - mean_test)^2 , na.rm=TRUE ) ) * sqrt( sum( (prediction - mean_pred)^2 , na.rm=TRUE ) ) )

			print(sprintf("Performance on fold %i: MSE=%.4f, R^2=%.4f, Rp=%.4f.", f,MSE,R2,Rp))
			MSEs = c(MSEs,MSE)
			R2s = c(R2s,R2)
			Rps = c(Rps,Rp)
		}

		average_MSE = mean(unlist(MSEs))
		all_MSEs = c(all_MSEs,average_MSE)
		average_R2 = mean(unlist(R2s))
		all_R2s = c(all_R2s,average_R2)
		average_Rp = mean(unlist(Rps))
		all_Rps = c(all_Rps,average_Rp)
		print(sprintf("Average performances for R=%i: MSE=%.4f, R^2=%.4f, Rp=%.4f.", R,average_MSE,average_R2,average_Rp))
	}

	# Find the best value for R, and return it
	best_R = R_values[[which.min(all_MSEs)]]
	print(sprintf("Best performance achieved with R=%i.", best_R))
	return(list(best_R, unlist(all_MSEs), unlist(all_R2s), unlist(all_Rps)))
}


