import jax.numpy as jnp
import jax
from tqdm import tqdm


def midpoints(start, end, num):
    """
    Compute midpoints of bins
    """
    rng = (end - start) / num
    return jnp.linspace(start + rng / 2, end - rng / 2, num)


def vmap_chunked1d(func, chunk_size=1024):
    """
    Vectorized map with chunking to reduce memory usage
    """

    vf = jax.jit(jax.vmap(func))

    def wrapper(arg):
        n = arg.shape[0]
        num_chunks = (n + chunk_size - 1) // chunk_size
        results = []
        for i in tqdm(range(0, n, chunk_size)):
            results.append(vf(arg[i : i + chunk_size]))
        return jnp.concatenate(results, axis=0)

    return wrapper
