import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm


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
        accept = jnp.log(u) < log_alpha  # (C,)

        # Update all three pieces in-place where accepted
        x = x.at[accept].set(x_prop[accept])  # (C, D)
        logp_cur = logp_cur.at[accept].set(logp_prop[accept])  # (C,)
        grad_cur = grad_cur.at[accept].set(grad_prop[accept])  # (C, D)

        samples = samples.at[t].set(x)  # (N, C, D)

        if progress:
            it.set_postfix(acc_rate=f"{accept.mean():0.2f}")

    return samples
