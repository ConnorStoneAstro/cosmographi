import jax.numpy as jnp

from .helpers import cdist


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


def gaussian_kernel(r):
    return jnp.exp(-0.5 * r**2)


def RBF_weights(X, y, scale, kernel=gaussian_kernel, s=0.0, degree=0):
    """
    Initialize Radial Basis Function interpolation to data (X, y) with length
    scale given by scale.

    Args:
        X: Input data points, shape (N, D)
        y: Output data points, shape (N,)
        scale: Length scale for each dimension, shape (D,)
        kernel: Kernel function, e.g. gaussian_kernel
        s: Regularization parameter to add to the diagonal of the kernel matrix
    """
    dists = cdist(X / scale, X / scale)
    K = kernel(dists) + s * jnp.eye(X.shape[0])
    if degree >= 0:
        P = jnp.hstack(
            [jnp.ones((X.shape[0], 1))] + [(X / scale) ** (i + 1) for i in range(degree)]
        )
        K = jnp.block([[K, P], [P.T, jnp.zeros((P.shape[1], P.shape[1]))]])
        y = jnp.hstack([y, jnp.zeros(P.shape[1])])
    weights = jnp.linalg.solve(K, y)
    return weights


def RBF_init(X, y, scale, kernel=gaussian_kernel, s=0.0, degree=0):
    weights = RBF_weights(X, y, scale, kernel=kernel, s=s, degree=degree)
    return dict(X=X, weights=weights, scale=scale, kernel=kernel, degree=degree)


def RBF(x, X, weights, scale, kernel=gaussian_kernel, degree=0):
    """
    Perform Radial Basis Function interpolation to data (X, y) with length scale
    given by scale, and evaluate the resulting model at points x.

    Args:
        x: Point to evaluate the RBF model at, shape (D,)
        X: Input data points, shape (N, D)
        w: Weights from RBF_init, shape (N,)
        scale: Length scale for each dimension, shape (D,)
        kernel: Kernel function, e.g. gaussian_kernel
    """
    dists = kernel(jnp.sqrt(((x[None, :] / scale - X / scale) ** 2).sum(-1)))
    if degree >= 0:
        p = jnp.hstack([1.0] + [(x / scale) ** (i + 1) for i in range(degree)])
        dists = jnp.hstack([dists, p])
    return jnp.dot(dists, weights)
