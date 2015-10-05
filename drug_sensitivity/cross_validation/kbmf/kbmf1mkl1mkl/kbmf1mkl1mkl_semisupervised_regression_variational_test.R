# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1mkl1mkl_semisupervised_regression_variational_test <- function(Kx, Kz, state) {
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Nz <- dim(Kz)[2]
  Pz <- dim(Kz)[3]
  R <- dim(state$Ax$mu)[2]

  Gx <- list(mu = array(0, c(R, Nx, Px)))
  for (m in 1:Px) {
    Gx$mu[,,m] <- crossprod(state$Ax$mu, Kx[,,m])
  }
  Hx <- list(mu = matrix(0, R, Nx))
  for (m in 1:Px) {
    Hx$mu <- Hx$mu + state$ex$mu[m] * Gx$mu[,,m]
  }

  Gz <- list(mu = array(0, c(R, Nz, Pz)))
  for (n in 1:Pz) {
    Gz$mu[,,n] <- crossprod(state$Az$mu, Kz[,,n])
  }
  Hz <- list(mu = matrix(0, R, Nz))
  for (n in 1:Pz) {
    Hz$mu <- Hz$mu + state$ez$mu[n] * Gz$mu[,,n]
  }

  Y <- list(mu = crossprod(Hx$mu, Hz$mu))

  prediction <- list(Gx = Gx, Hx = Hx, Gz = Gz, Hz = Hz, Y = Y)
}