import jax.numpy as jnp
import jax
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from itertools import combinations_with_replacement


def midpoints(start, end, num):
    """
    Compute midpoints of bins
    """
    rng = (end - start) / num
    return jnp.linspace(start + rng / 2, end - rng / 2, num)


def vmap_chunked1d(func, chunk_size=1024, prog_bar=False):
    """
    Vectorized map with chunking to reduce memory usage
    """

    vf = jax.jit(jax.vmap(func))

    def wrapper(arg):
        n = arg.shape[0]
        num_chunks = (n + chunk_size - 1) // chunk_size
        results = []
        for i in tqdm(range(0, n, chunk_size), disable=not prog_bar):
            results.append(vf(arg[i : i + chunk_size]))
        return jnp.concatenate(results, axis=0)

    return wrapper


def int_Phi_N(mu1, sigma1, mu2, sigma2):
    # int_{-inf}^{inf} Phi((x - mu1) / sigma1) N(x|mu2, sigma2^2) dx
    # where Phi is the CDF of the standard normal distribution
    return jax.scipy.stats.norm.logcdf(-(mu1 - mu2) / jnp.sqrt(sigma1**2 + sigma2**2))


@jax.jit
def cdist_pbc(x, y):
    return jnp.sqrt(jnp.sum(((x[:, None] - y[None, :] + 1) % 2 - 1) ** 2, -1))


@jax.jit
def cdist(x, y):
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


def _polynomial_features(d, degree):
    pows = []
    for deg in range(degree + 1):
        pows.extend(combinations_with_replacement(range(d), deg))
    pows = jnp.array([[i.count(j) for j in range(d)] for i in pows], dtype=jnp.int32)
    return pows


def _polynomial_transform(X, pows):
    return jnp.prod(X[:, None] ** pows[None], axis=-1)


def tdp_regression(X, y, degree):
    """
    Perform polynomial regression with all cross terms up to a given total
    degree.

    First performs X -> X_poly by converting the input features to all
    polynomial features up to the given total degree, including cross terms.
    This means for example in 2 feature dimensions going up to degree d would
    mean all `X0^n * X1^m` where n + m <= d. Then performs linear regression on
    the transformed features to fit the target values y. Uses
    linalg.lstsq(X_poly, y) to solve for the coefficients.

    Parameters: X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The target values.
    degree : int
        The total degree of the polynomial features.

    Returns: coeffs : array, shape (n_output_features,)
        The coefficients of the polynomial regression model.
    poly : array, shape (n_output_features, n_features)
        The polynomial feature transformation.
    """

    poly = _polynomial_features(X.shape[1], degree)
    X_poly = _polynomial_transform(X, poly)

    coefs = jnp.linalg.lstsq(X_poly, y)[0]

    return coefs, poly


def tdp_evaluate(X, coefs, poly):
    """
    Evaluate the polynomial regression model at new data points.
    See `tdp_regression` for details.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    coefs : array, shape (n_output_features,)
        The coefficients of the polynomial regression model.
    poly : array, shape (n_output_features, n_features)
        The polynomial feature transformation.

    Returns:
    y_pred : array, shape (n_samples,)
        The predicted values.
    """
    X_poly = _polynomial_transform(X, poly)
    return jnp.dot(X_poly, coefs)
