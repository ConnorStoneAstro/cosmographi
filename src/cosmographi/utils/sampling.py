import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm
import numpy as np
from scipy.stats.qmc import LatinHypercube


from .helpers import cdist_pbc


def mala(
    initial_state,  # (num_chains, D)
    log_prob,  # f(x) -> logP ()
    num_samples,
    epsilon,
    mass_matrix=None,  # covariance
    progress=True,
    desc="MALA",
    key=None,
):
    x = jnp.array(initial_state, copy=True)
    C, D = x.shape
    vlogP = jax.jit(jax.vmap(log_prob))
    vlogP_grad = jax.jit(jax.vmap(jax.grad(log_prob)))

    # mass, inv_mass, L
    if mass_matrix is None:
        mass = jnp.eye(D)  # (D, D)
    else:
        mass = jnp.array(mass_matrix, copy=True)  # (D, D)
    inv_mass = jnp.linalg.inv(mass)  # (D, D)
    L = jnp.linalg.cholesky(mass)  # (D, D)

    samples = jnp.zeros((num_samples, C, D), dtype=x.dtype)  # (N, C, D)
    accept = jnp.zeros((num_samples, C), dtype=bool)  # (N,C)

    # Cache current state
    logp_cur = vlogP(x)  # (C,)
    grad_cur = vlogP_grad(x)  # (C, D)

    # Random number generator
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(1e10))

    it = range(num_samples)
    if progress:
        it = tqdm(it, desc=desc, position=0, leave=True)

    for t in it:
        # proposal using current grad
        mu_x = 0.5 * (epsilon**2) * (grad_cur @ mass)  # (C, D)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (C, D)) @ L.T  # (C, D)
        x_prop = x + mu_x + epsilon * noise  # (C, D)

        # Evaluate proposal
        logp_prop = vlogP(x_prop)  # (C,)
        grad_prop = vlogP_grad(x_prop)  # (C, D)

        mu_xprop = 0.5 * (epsilon**2) * (grad_prop @ mass)  # (C, D)

        # q(x|x') \propto \exp(-0.5|x - x' - mu(x')|^2 / \epsilon^2)
        d1 = x - x_prop - mu_xprop  # for q(x | x')
        d2 = x_prop - x - mu_x  # for q(x'| x)

        logq1 = -0.5 * jnp.einsum("bi,ij,bj->b", d1, inv_mass, d1) / epsilon**2  # (C,)
        logq2 = -0.5 * jnp.einsum("bi,ij,bj->b", d2, inv_mass, d2) / epsilon**2  # (C,)

        log_alpha = (logp_prop - logp_cur) + (logq1 - logq2)  # (C,)

        # Accept or reject
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, (C,))  # (C,)
        accept = accept.at[t].set(jnp.log(u) < log_alpha)  # (N, C)

        # Update all three pieces in-place where accepted
        x = x.at[accept[t]].set(x_prop[accept[t]])  # (C, D)
        logp_cur = logp_cur.at[accept[t]].set(logp_prop[accept[t]])  # (C,)
        grad_cur = grad_cur.at[accept[t]].set(grad_prop[accept[t]])  # (C, D)

        samples = samples.at[t].set(x)  # (N, C, D)

        if progress:
            it.set_postfix(acc_rate=f"{accept[: t + 1].mean():0.2f}")

    return samples


def superuniform(key, n, d=1, c=10, bounds=None):
    """
    Samples `n` points in `d` dimensions in the hypercube [-1, 1]^d using a
    superuniform sampling algorithm. The algorithm works by iteratively adding
    points that are far away from existing points, ensuring a more uniform
    distribution than simple random sampling.

    The algorithm works as follows::

        1. Start with a single uniform random sample.
        2. For each subsequent point, generate `c` candidate points uniformly at
        random.
        3. For each candidate, compute the periodic boundary distance to the nearest existing point.
        4. Select the candidate that maximizes the distance to the nearest point.
        5. Repeat until `n` points are sampled.

    Args:
        key: JAX random key for random number generation.
        n (int or tuple): Number of points to sample. If a tuple (m, n) is
            provided, the function returns `m` sets of `n` points each.
        d (int): Dimensionality of the hypercube.
        c (int): Number of candidate points to consider for each new point.
        bounds (tuple): Optional length (2, d) tuple specifying the (min, max) bounds for
            rescaling the points from [-1, 1] to [min, max].
    """
    if isinstance(n, tuple) and len(n) == 2:
        subkeys = jax.random.split(key, n[0])
        return jax.vmap(superuniform, in_axes=(0, None, None, None, None))(
            subkeys, n[1], d, c, bounds
        )
    x = jnp.zeros(shape=(n, d))
    key, subkey = jax.random.split(key)
    x = x.at[0].set(jax.random.uniform(subkey, shape=(d,)) * 2 - 1)

    for i in range(1, n):
        key, subkey = jax.random.split(key)
        candidates = jax.random.uniform(subkey, shape=(c, d)) * 2 - 1

        D = cdist_pbc(x[:i], candidates)

        x = x.at[i].set(candidates[jnp.argmax(jnp.min(D, axis=0))])

    if bounds is not None:
        bounds = jnp.array(bounds)
        x = 0.5 * (x + 1) * (bounds[1] - bounds[0]) + bounds[0]
    return x


def latin_hypercube(m, n, d, bounds=None, seed=None):
    """
    Run scipy Latin Hypercube sampling m times for samples of n points in d dimensions.

    Args:
        m: number of random sets to draw
        n: number of points in a random set
        d: dimensionality of the hypercube
        bounds (optional): length (2,d) np.array giving (min, max) bounds of hyper-cube
        seed: int to pass to np.random.default_rng to make random number generator
    """

    rng = np.random.default_rng(seed)
    sampler = LatinHypercube(d=d, rng=rng)
    lhs = list(sampler.random(n=n) for _ in range(m))
    lhs = np.stack(lhs)

    if bounds is not None:
        lhs = lhs * (bounds[1] - bounds[0]) + bounds[0]
    return jnp.array(lhs)
