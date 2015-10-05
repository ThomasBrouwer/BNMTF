# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1mkl1mkl_supervised_classification_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  Pz <- dim(Kz)[3]
  R <- parameters$R
  sigma_g <- parameters$sigma_g
  sigma_h <- parameters$sigma_h

  Lambdax <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dx, R), beta = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mu = matrix(rnorm(Dx * R), Dx, R), sigma = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mu = array(rnorm(R * Nx * Px), c(R, Nx, Px)), sigma = array(diag(1, R, R), c(R, R, Px)))
  etax <- list(alpha = matrix(parameters$alpha_eta + 0.5, Px, 1), beta = matrix(parameters$beta_eta, Px, 1))
  ex <- list(mu = matrix(1, Px, 1), sigma = diag(1, Px, Px))
  Hx <- list(mu = matrix(rnorm(R * Nx), R, Nx), sigma = diag(1, R, R))

  Lambdaz <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dz, R), beta = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mu = matrix(rnorm(Dz * R), Dz, R), sigma = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mu = array(rnorm(R * Nz * Pz), c(R, Nz, Pz)), sigma = array(diag(1, R, R), c(R, R, Pz)))
  etaz <- list(alpha = matrix(parameters$alpha_eta + 0.5, Pz, 1), beta = matrix(parameters$beta_eta, Pz, 1))
  ez <- list(mu = matrix(1, Pz, 1), sigma = diag(1, Pz, Pz))
  Hz <- list(mu = matrix(rnorm(R * Nz), R, Nz), sigma = diag(1, R, R))

  F <- list(mu = (abs(matrix(rnorm(Nx * Nz), Nx, Nz)) + parameters$margin) * sign(Y), sigma = matrix(1, Nx, Nz))

  KxKx <- matrix(0, Dx, Dx)
  for (m in 1:Px) {
    KxKx <- KxKx + tcrossprod(Kx[,,m], Kx[,,m])
  }
  Kx <- matrix(Kx, Dx, Nx * Px)

  KzKz <- matrix(0, Dz, Dz)
  for (n in 1:Pz) {
    KzKz <- KzKz + tcrossprod(Kz[,,n], Kz[,,n])
  }
  Kz <- matrix(Kz, Dz, Nz * Pz)

  lower <- matrix(-1e40, Nx, Nz)
  lower[which(Y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, Nx, Nz)
  upper[which(Y < 0)] <- -parameters$margin

  for (iter in 1:parameters$iteration) {
    # update Lambdax
    for (s in 1:R) {
      Lambdax$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Ax$mu[,s]^2 + diag(Ax$sigma[,,s])))
    }
    # update Ax
    for (s in 1:R) {
      Ax$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambdax$alpha[,s] * Lambdax$beta[,s]), Dx, Dx) + KxKx / sigma_g^2))
      Ax$mu[,s] <- Ax$sigma[,,s] %*% (Kx %*% matrix(Gx$mu[s,,], Nx * Px, 1) / sigma_g^2)
    }
    # update Gx
    for (m in 1:Px) {
      Gx$sigma[,,m] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + diag((ex$mu[m] * ex$mu[m] + ex$sigma[m, m]) / sigma_h^2, R, R)))
      Gx$mu[,,m] <- crossprod(Ax$mu, Kx[,((m - 1) * Nx + 1):(m * Nx)]) / sigma_g^2 + ex$mu[m] * Hx$mu / sigma_h^2
      for (o in setdiff(1:Px, m)) {
        Gx$mu[,,m] <- Gx$mu[,,m] - (ex$mu[m] * ex$mu[o] + ex$sigma[m, o]) * Gx$mu[,,o] / sigma_h^2
      }
      Gx$mu[,,m] <- Gx$sigma[,,m] %*% Gx$mu[,,m]
    }
    # update etax
    etax$beta <- 1 / (1 / parameters$beta_eta + 0.5 * (ex$mu^2 + diag(ex$sigma)))
    # update ex
    ex$sigma <- diag(as.vector(etax$alpha * etax$beta))
    for (m in 1:Px) {
      for (o in 1:Px) {
        ex$sigma[m, o] <- ex$sigma[m, o] + (sum(Gx$mu[,,m] * Gx$mu[,,o]) + (m == o) * Nx * sum(diag(Gx$sigma[,,m]))) / sigma_h^2
      }
    }
    ex$sigma <- chol2inv(chol(ex$sigma))
    for (m in 1:Px) {
      ex$mu[m] <- sum(Gx$mu[,,m] * Hx$mu) / sigma_h^2
    }
    ex$mu <- ex$sigma %*% ex$mu
    # update Hx
    Hx$sigma <- chol2inv(chol(diag(1 / sigma_h^2, R, R) + tcrossprod(Hz$mu, Hz$mu) + Nz * Hz$sigma))
    Hx$mu <- tcrossprod(Hz$mu, F$mu)
    for (m in 1:Px) {
      Hx$mu <- Hx$mu + ex$mu[m] * Gx$mu[,,m] / sigma_h^2
    }
    Hx$mu <- Hx$sigma %*% Hx$mu

    # update Lambdaz
    for (s in 1:R) {
      Lambdaz$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Az$mu[,s]^2 + diag(Az$sigma[,,s])))
    }
    # update Az
    for (s in 1:R) {
      Az$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambdaz$alpha[,s] * Lambdaz$beta[,s]), Dz, Dz) + KzKz / sigma_g^2))
      Az$mu[,s] <- Az$sigma[,,s] %*% (Kz %*% matrix(Gz$mu[s,,], Nz * Pz, 1) / sigma_g^2)
    }
    # update Gz
    for (n in 1:Pz) {
      Gz$sigma[,,n] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + diag((ez$mu[n] * ez$mu[n] + ez$sigma[n, n]) / sigma_h^2, R, R)))
      Gz$mu[,,n] <- crossprod(Az$mu, Kz[,((n - 1) * Nz + 1):(n * Nz)]) / sigma_g^2 + ez$mu[n] * Hz$mu / sigma_h^2
      for (p in setdiff(1:Pz, n)) {
        Gz$mu[,,n] <- Gz$mu[,,n] - (ez$mu[n] * ez$mu[p] + ez$sigma[n, p]) * Gz$mu[,,p] / sigma_h^2
      }
      Gz$mu[,,n] <- Gz$sigma[,,n] %*% Gz$mu[,,n]
    }
    # update etaz
    etaz$beta <- 1 / (1 / parameters$beta_eta + 0.5 * (ez$mu^2 + diag(ez$sigma)))
    # update ez
    ez$sigma <- diag(as.vector(etaz$alpha * etaz$beta))
    for (n in 1:Pz) {
      for (p in 1:Pz) {
        ez$sigma[n, p] <- ez$sigma[n, p] + (sum(Gz$mu[,,n] * Gz$mu[,,p]) + (n == p) * Nz * sum(diag(Gz$sigma[,,n]))) / sigma_h^2
      }
    }
    ez$sigma <- chol2inv(chol(ez$sigma))
    for (n in 1:Pz) {
      ez$mu[n] <- sum(Gz$mu[,,n] * Hz$mu) / sigma_h^2
    }
    ez$mu <- ez$sigma %*% ez$mu
    # update Hz
    Hz$sigma <- chol2inv(chol(diag(1 / sigma_h^2, R, R) + tcrossprod(Hx$mu, Hx$mu) + Nx * Hx$sigma))
    Hz$mu <- Hx$mu %*% F$mu
    for (n in 1:Pz) {
      Hz$mu <- Hz$mu + ez$mu[n] * Gz$mu[,,n] / sigma_h^2
    }
    Hz$mu <- Hz$sigma %*% Hz$mu

    # update F
    output <- crossprod(Hx$mu, Hz$mu)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, etax = etax, ex = ex, Lambdaz = Lambdaz, Az = Az, etaz = etaz, ez = ez, parameters = parameters)
}