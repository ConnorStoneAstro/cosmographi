import jax.numpy as jnp


def midpoints(start, end, num):
    """
    Compute midpoints of bins
    """
    rng = (end - start) / num
    return jnp.linspace(start + rng / 2, end - rng / 2, num)
