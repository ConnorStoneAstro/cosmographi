import jax
from scipy.special import roots_legendre
import jax.numpy as jnp
from functools import lru_cache
from .helpers import midpoints


def mid(f, a, b, n=100):
    x = midpoints(a, b, n)
    return jnp.mean(f(x)) * (b - a)


@lru_cache(maxsize=32)
def _quad_table(n=100):
    """Compute abscissas and weights for Gauss-Legendre quadrature.

    Args:
        n: Number of points to use in the integration.

    Returns:
        abscissas and weights for Gauss-Legendre quadrature.
    """
    return roots_legendre(n)


def quad(f, a, b, n=100, args=()):
    """Numerical integration using Gauss-Legendre quadrature.

    Args:
        func: Function to integrate. Must be a function of a single variable.
        a: Lower limit of integration.
        b: Upper limit of integration.
        n: Number of points to use in the integration.

    Returns:
        Approximation of the integral of func from a to b.
    """
    abscissa, weights = _quad_table(n)
    x = (abscissa + 1.0) * (b - a) / 2.0 + a
    y = f(x, *args)
    return jnp.sum(weights * y) * (b - a) / 2.0


def log_quad(log_f, a, b, n=100, args=()):
    """Numerical integration of a log function using Gauss-Legendre quadrature.

    Args:
        log_f: Log of the function to integrate. Must be a function of a single variable.
        a: Lower limit of integration.
        b: Upper limit of integration.
        n: Number of points to use in the integration.

    Returns:
        Approximation of the integral of exp(log_f) from a to b.
    """
    abscissa, weights = _quad_table(n)
    x = (abscissa + 1.0) * (b - a) / 2.0 + a
    log_y = log_f(x, *args)
    return jax.nn.logsumexp(jnp.log(weights) + log_y) + jnp.log((b - a) / 2.0)


def gauss_rescale_integrate(f, a, b, mu, sigma, n=100, args=()):
    """
    integrate f from a to b using a rescaled coordinate y = invCDF(x) of a Gaussian described by mu and sigma
    """

    def x(y):
        return jax.scipy.stats.norm.ppf(y, loc=mu, scale=sigma)

    def y(x):
        return jax.scipy.stats.norm.cdf(x, loc=mu, scale=sigma)

    def integrand(y):
        return f(x(y), *args) / jax.scipy.stats.norm.pdf(x(y), loc=mu, scale=sigma)

    return quad(integrand, y(a), y(b), n=n)


def log_gauss_rescale_integrate(log_f, a, b, mu, sigma, n=100, args=()):
    """
    integrate f from a to b using a rescaled coordinate y = invCDF(x) of a Gaussian described by mu and sigma
    """

    def x(y):
        return jax.scipy.stats.norm.ppf(y, loc=mu, scale=sigma)

    def y(x):
        return jax.scipy.stats.norm.cdf(x, loc=mu, scale=sigma)

    def integrand(y):
        return log_f(x(y), *args) - jax.scipy.stats.norm.logpdf(x(y), loc=mu, scale=sigma)

    return log_quad(integrand, y(a), y(b), n=n)
