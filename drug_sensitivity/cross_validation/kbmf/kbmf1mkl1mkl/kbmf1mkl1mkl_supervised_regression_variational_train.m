% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmf1mkl1mkl_supervised_regression_variational_train(Kx, Kz, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    Dx = size(Kx, 1);
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Dz = size(Kz, 1);
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = parameters.R;
    sigma_g = parameters.sigma_g;
    sigma_h = parameters.sigma_h;
    sigma_y = parameters.sigma_y;

    Lambdax.alpha = (parameters.alpha_lambda + 0.5) * ones(Dx, R);
    Lambdax.beta = parameters.beta_lambda * ones(Dx, R);
    Ax.mu = randn(Dx, R);
    Ax.sigma = repmat(eye(Dx, Dx), [1, 1, R]);
    Gx.mu = randn(R, Nx, Px);
    Gx.sigma = repmat(eye(R, R), [1, 1, Px]);
    etax.alpha = (parameters.alpha_eta + 0.5) * ones(Px, 1);
    etax.beta = parameters.beta_eta * ones(Px, 1);
    ex.mu = ones(Px, 1);
    ex.sigma = eye(Px, Px);
    Hx.mu = randn(R, Nx);
    Hx.sigma = eye(R, R);

    Lambdaz.alpha = (parameters.alpha_lambda + 0.5) * ones(Dz, R);
    Lambdaz.beta = parameters.beta_lambda * ones(Dz, R);
    Az.mu = randn(Dz, R);
    Az.sigma = repmat(eye(Dz, Dz), [1, 1, R]);
    Gz.mu = randn(R, Nz, Pz);
    Gz.sigma = repmat(eye(R, R), [1, 1, Pz]);
    etaz.alpha = (parameters.alpha_eta + 0.5) * ones(Pz, 1);
    etaz.beta = parameters.beta_eta * ones(Pz, 1);
    ez.mu = ones(Pz, 1);
    ez.sigma = eye(Pz, Pz);
    Hz.mu = randn(R, Nz);
    Hz.sigma = eye(R, R);

    KxKx = zeros(Dx, Dx);
    for m = 1:Px
        KxKx = KxKx + Kx(:, :, m) * Kx(:, :, m)';
    end
    Kx = reshape(Kx, [Dx, Nx * Px]);

    KzKz = zeros(Dz, Dz);
    for n = 1:Pz
        KzKz = KzKz + Kz(:, :, n) * Kz(:, :, n)';
    end
    Kz = reshape(Kz, [Dz, Nz * Pz]);

    lambdax_indices = repmat(logical(eye(Dx, Dx)), [1, 1, R]);
    lambdaz_indices = repmat(logical(eye(Dz, Dz)), [1, 1, R]);

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Lambdax
        Lambdax.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Ax.mu.^2 + reshape(Ax.sigma(lambdax_indices), Dx, R)));

        %%%% update Ax
        for s = 1:R
            Ax.sigma(:, :, s) = (diag(Lambdax.alpha(:, s) .* Lambdax.beta(:, s)) + KxKx / sigma_g^2) \ eye(Dx, Dx);
            Ax.mu(:, s) = Ax.sigma(:, :, s) * (Kx * reshape(squeeze(Gx.mu(s, :, :)), Nx * Px, 1) / sigma_g^2);
        end

        %%%% update Gx
        for m = 1:Px
            Gx.sigma(:, :, m) = (eye(R, R) / sigma_g^2 + (ex.mu(m) * ex.mu(m) + ex.sigma(m, m)) * eye(R, R) / sigma_h^2) \ eye(R, R);
            Gx.mu(:, :, m) = Ax.mu' * Kx(:, (m - 1) * Nx + 1:m * Nx) / sigma_g^2 + ex.mu(m) * Hx.mu / sigma_h^2;
            for o = [1:m - 1, m + 1:Px]
                Gx.mu(:, :, m) = Gx.mu(:, :, m) - (ex.mu(m) * ex.mu(o) + ex.sigma(m, o)) * Gx.mu(:, :, o) / sigma_h^2;
            end
            Gx.mu(:, :, m) = Gx.sigma(:, :, m) * Gx.mu(:, :, m);
        end

        %%%% update etax
        etax.beta = 1 ./ (1 / parameters.beta_eta + 0.5 * (ex.mu.^2 + diag(ex.sigma)));

        %%%% update ex
        ex.sigma = diag(etax.alpha .* etax.beta);
        for m = 1:Px
            for o = 1:Px
                ex.sigma(m, o) = ex.sigma(m, o) + (sum(sum(Gx.mu(:, :, m) .* Gx.mu(:, :, o))) + (m == o) * Nx * sum(diag(Gx.sigma(:, :, m)))) / sigma_h^2;
            end
        end
        ex.sigma = ex.sigma \ eye(Px, Px);
        for m = 1:Px
            ex.mu(m) = sum(sum(Gx.mu(:, :, m) .* Hx.mu)) / sigma_h^2;
        end
        ex.mu = ex.sigma * ex.mu;

        %%%% update Hx
        Hx.sigma = (eye(R, R) / sigma_h^2 + (Hz.mu * Hz.mu' + Nz * Hz.sigma) / sigma_y^2) \ eye(R, R);
        Hx.mu = Hz.mu * Y' / sigma_y^2;
        for m = 1:Px
            Hx.mu = Hx.mu + ex.mu(m) * Gx.mu(:, :, m) / sigma_h^2;
        end
        Hx.mu = Hx.sigma * Hx.mu;

        %%%% update Lambdaz
        Lambdaz.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Az.mu.^2 + reshape(Az.sigma(lambdaz_indices), Dz, R)));

        %%%% update Az
        for s = 1:R
            Az.sigma(:, :, s) = (diag(Lambdaz.alpha(:, s) .* Lambdaz.beta(:, s)) + KzKz / sigma_g^2) \ eye(Dz, Dz);
            Az.mu(:, s) = Az.sigma(:, :, s) * (Kz * reshape(squeeze(Gz.mu(s, :, :)), Nz * Pz, 1) / sigma_g^2);
        end

        %%%% update Gz
        for n = 1:Pz
            Gz.sigma(:, :, n) = (eye(R, R) / sigma_g^2 + (ez.mu(n) * ez.mu(n) + ez.sigma(n, n)) * eye(R, R) / sigma_h^2) \ eye(R, R);
            Gz.mu(:, :, n) = Az.mu' * Kz(:, (n - 1) * Nz + 1:n * Nz) / sigma_g^2 + ez.mu(n) * Hz.mu / sigma_h^2;
            for p = [1:n - 1, n + 1:Pz]
                Gz.mu(:, :, n) = Gz.mu(:, :, n) - (ez.mu(n) * ez.mu(p) + ez.sigma(n, p)) * Gz.mu(:, :, p) / sigma_h^2;
            end
            Gz.mu(:, :, n) = Gz.sigma(:, :, n) * Gz.mu(:, :, n);
        end

        %%%% update etaz
        etaz.beta = 1 ./ (1 / parameters.beta_eta + 0.5 * (ez.mu.^2 + diag(ez.sigma)));

        %%%% update ez
        ez.sigma = diag(etaz.alpha .* etaz.beta);
        for n = 1:Pz
            for p = 1:Pz
                ez.sigma(n, p) = ez.sigma(n, p) + (sum(sum(Gz.mu(:, :, n) .* Gz.mu(:, :, p))) + (n == p) * Nz * sum(diag(Gz.sigma(:, :, n)))) / sigma_h^2;
            end
        end
        ez.sigma = ez.sigma \ eye(Pz, Pz);
        for n = 1:Pz
            ez.mu(n) = sum(sum(Gz.mu(:, :, n) .* Hz.mu)) / sigma_h^2;
        end
        ez.mu = ez.sigma * ez.mu;

        %%%% update Hz
        Hz.sigma = (eye(R, R) / sigma_h^2 + (Hx.mu * Hx.mu' + Nx * Hx.sigma) / sigma_y^2) \ eye(R, R);
        Hz.mu = Hx.mu * Y / sigma_y^2;
        for n = 1:Pz
            Hz.mu = Hz.mu + ez.mu(n) * Gz.mu(:, :, n) / sigma_h^2;
        end
        Hz.mu = Hz.sigma * Hz.mu;
    end

    state.Lambdax = Lambdax;
    state.Ax = Ax;
    state.etax = etax;
    state.ex = ex;
    state.Lambdaz = Lambdaz;
    state.Az = Az;
    state.etaz = etaz;
    state.ez = ez;
    state.parameters = parameters;
end