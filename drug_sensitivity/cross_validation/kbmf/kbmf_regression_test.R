# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf_regression_test <- function(Kx, Kz, state) {
  prediction <- state$parameters$test_function(drop(Kx), drop(Kz), state)
}