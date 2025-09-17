from scipy.special import roots_legendre
import jax.numpy as jnp
from functools import lru_cache


@lru_cache(maxsize=32)
def _quad_table(n=100):
    """Compute abscissas and weights for Gauss-Legendre quadrature.

    Args:
        n: Number of points to use in the integration.

    Returns:
        abscissas and weights for Gauss-Legendre quadrature.
    """
    return roots_legendre(n)


def quad(func, a, b, n=100):
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
    y = func(x)
    return jnp.sum(weights * y) * (b - a) / 2.0
