% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf1mkl1mkl_supervised_regression_variational_test(Kx, Kz, state)
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = size(state.Ax.mu, 2);

    prediction.Gx.mu = zeros(R, Nx, Px);
    for m = 1:Px
        prediction.Gx.mu(:, :, m) = state.Ax.mu' * Kx(:, :, m);
    end
    prediction.Hx.mu = zeros(R, Nx);
    for m = 1:Px
        prediction.Hx.mu = prediction.Hx.mu + state.ex.mu(m) * prediction.Gx.mu(:, :, m);
    end

    prediction.Gz.mu = zeros(R, Nz, Pz);
    for n = 1:Pz
        prediction.Gz.mu(:, :, n) = state.Az.mu' * Kz(:, :, n);
    end
    prediction.Hz.mu = zeros(R, Nz);
    for n = 1:Pz
        prediction.Hz.mu = prediction.Hz.mu + state.ez.mu(n) * prediction.Gz.mu(:, :, n);
    end

    prediction.Y.mu = prediction.Hx.mu' * prediction.Hz.mu;
end