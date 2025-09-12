import numpy as np


def set_seed(s):
    np.random.seed(s)


# sample W uniformly on the unit sphere in R^d
def sample_sphere(m, d):
    Z = np.random.randn(m, d)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z


def relu(z):
    return np.maximum(z, 0.0)


# account for biases and compute activations
def make_features(X, W):
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return relu(Xa @ W.T)


# exact solution of u* = argmin_u ||Phi u - y||^2
def solve_u_star(Phi, y, reg=1e-8):
    A = Phi.T @ Phi
    if reg > 0:
        A = A + reg * np.eye(A.shape[0])
    return np.linalg.solve(A, Phi.T @ y)


def mse(yhat, y):
    r = yhat - y
    return float(np.mean(r * r))
