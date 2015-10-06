# Mehmet Gonen (mehmet.gonen@gmail.com)

source_directory <- function(path) {
  files <- sort(dir(path, "\\.[rR]$", full.names = TRUE))
  lapply(files, source, chdir = TRUE)
}

kbmf_regression_train <- function(Kx, Kz, Y, R, varargin) {
  source_directory("kbmf1k1k")
  source_directory("kbmf1k1mkl")
  source_directory("kbmf1mkl1k")
  source_directory("kbmf1mkl1mkl")
  Px <- dim(Kx)[3]
  Pz <- dim(Kz)[3]
  is_supervised <- all(!is.na(Y))

  parameters <- list()
  parameters$alpha_lambda <- 1
  parameters$beta_lambda <- 1
  if (Px > 1 || Pz > 1) {
    parameters$alpha_eta <- 1
    parameters$beta_eta <- 1
  }
  parameters$iteration <- 200
  parameters$progress <- 1
  parameters$R <- R
  parameters$seed <- 1606
  parameters$sigma_g <- 0.1
  if (Px > 1 || Pz > 1) {
    parameters$sigma_h <- 0.1
  }
  parameters$sigma_y <- 1

  if (is_supervised == 1) {
    if (Px == 1 && Pz == 1) {
      train_function <- kbmf1k1k_supervised_regression_variational_train
      test_function <- kbmf1k1k_supervised_regression_variational_test  
    }
    if (Px > 1 && Pz == 1) {
      train_function <- kbmf1mkl1k_supervised_regression_variational_train
      test_function <- kbmf1mkl1k_supervised_regression_variational_test  
    }
    if (Px == 1 && Pz > 1) {
      train_function <- kbmf1k1mkl_supervised_regression_variational_train
      test_function <- kbmf1k1mkl_supervised_regression_variational_test  
    }
    if (Px > 1 && Pz > 1) {
      train_function <- kbmf1mkl1mkl_supervised_regression_variational_train
      test_function <- kbmf1mkl1mkl_supervised_regression_variational_test  
    }
  }
  else {
    if (Px == 1 && Pz == 1) {
      train_function <- kbmf1k1k_semisupervised_regression_variational_train
      test_function <- kbmf1k1k_semisupervised_regression_variational_test  
    }
    if (Px > 1 && Pz == 1) {
      train_function <- kbmf1mkl1k_semisupervised_regression_variational_train
      test_function <- kbmf1mkl1k_semisupervised_regression_variational_test  
    }
    if (Px == 1 && Pz > 1) {
      train_function <- kbmf1k1mkl_semisupervised_regression_variational_train
      test_function <- kbmf1k1mkl_semisupervised_regression_variational_test  
    }
    if (Px > 1 && Pz > 1) {
      train_function <- kbmf1mkl1mkl_semisupervised_regression_variational_train
      test_function <- kbmf1mkl1mkl_semisupervised_regression_variational_test  
    }
  }

  if (missing(varargin) == FALSE) {
    for (name in names(varargin)) {
      parameters[[name]] <- varargin[[name]]
    } 
  }

  parameters$train_function <- train_function
  parameters$test_function <- test_function

  state <- train_function(drop(Kx), drop(Kz), Y, parameters)
}
