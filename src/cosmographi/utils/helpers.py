import jax.numpy as jnp
import jax
from tqdm import tqdm


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
