import jax.numpy as jnp


def WLS(x, X, y, scale, order=1):
    """
    Perform weighted least squares (WLS) fit to data (X, y) with length
    scale on each dimension given by scale, and evaluate the resulting
    model at points x.

    Args:
        x: Point to evaluate the linear model at, shape (D,)
        X: Input data points, shape (N, D)
        y: Output data points, shape (N,)
        scale: Weights for each dimension, shape (D,)
    """
    W = jnp.diag(jnp.exp(-0.5 * jnp.sum(((X - x) / scale) ** 2, axis=-1)))

    X_aug = jnp.hstack([jnp.ones((X.shape[0], 1))] + [X ** (i + 1) for i in range(order)])
    A = X_aug.T @ W @ X_aug
    b = X_aug.T @ W @ y
    coeffs = jnp.linalg.solve(A, b)
    x_aug = jnp.hstack([1.0] + [x ** (i + 1) for i in range(order)])
    return x_aug @ coeffs


def RBF(x, X, y, scale):
    """
    Perform Radial Basis Function interpolation to data (X, y) with length scale
    given by scale, and evaluate the resulting model at points x.

    Args:
        x: Point to evaluate the RBF model at, shape (D,)
        X: Input data points, shape (N, D)
        y: Output data points, shape (N,)
        scale: Length scale for each dimension, shape (D,)
    """
    dists = jnp.sum(((X - x) / scale) ** 2, axis=-1)
    weights = jnp.exp(-0.5 * dists)
    return jnp.sum(weights * y) / jnp.sum(weights)
