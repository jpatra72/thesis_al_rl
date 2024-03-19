import numpy as np


def calculate_gp_posterior(x_train, y_train, x_test, kernel, noise, n_samples=3):
    # matrices
    k_xx = kernel(x_test, x_test)
    k_Xx = kernel(x_train, x_test)
    k_XX = kernel(x_train, x_train)

    # mean
    L = np.linalg.cholesky(k_XX + np.eye(k_XX.shape[0]) * noise **2)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    f_mean = (k_Xx.T @ alpha).reshape(-1)

    # cov
    v = np.linalg.solve(L, k_Xx)
    f_var = k_xx - v.T @ v

    # get samples
    samples = np.random.multivariate_normal(f_mean.reshape(-1), f_var, size=n_samples)
    return f_mean, f_var, samples