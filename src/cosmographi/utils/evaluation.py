import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multinomial, chi2 as chi2_dist
from scipy.optimize import root_scalar
import numpy as np


def multinomial_null_pvalue(
    key: jax.Array,
    c: jnp.ndarray,  # shape (N,), integer labels in {0, ..., M}
    M: int,
    T: int,
):
    """
    Monte-Carlo exact (probability-ordering) two-sided p-value that the
    observed label vector c (length N) arises from Uniform{0,...,M}.

    Returns:
      p_value: scalar in (0,1]
      logpmf_obs: scalar log pmf of observed counts under the null
      counts_obs: (K,) observed counts with K=M+1
      logpmf_sims: (T,) log pmfs of the simulated null tables
    """
    K = M + 1
    c = jnp.asarray(c, dtype=jnp.int32)
    N = c.shape[0]

    # Uniform null over K categories
    p = jnp.ones(K) / K

    # Observed counts and log-PMF under the null
    counts_obs = jnp.bincount(c, length=K)  # (K,)
    logpmf_obs = multinomial.logpmf(counts_obs, n=N, p=p)  # scalar

    # Simulate T null datasets (each N iid Uniform{0,...,M})
    sims = random.randint(key, shape=(T, N), minval=0, maxval=K)  # (T, N)
    counts_sims = jax.vmap(lambda row: jnp.bincount(row, length=K))(sims)  # (T, K)
    logpmf_sims = jax.vmap(lambda cnt: multinomial.logpmf(cnt, n=N, p=p))(counts_sims)  # (T,)

    # Probability-ordering (two-sided exact) p-value with +1/(T+1) adjustment
    num_extreme = 2 * jnp.min(
        jnp.array([jnp.sum(logpmf_sims < logpmf_obs), jnp.sum(logpmf_sims > logpmf_obs)])
    )
    p_value = (1.0 + num_extreme) / (T + 1.0)

    return p_value


def two_tailed_p(chi2, df):
    assert df > 2, "Degrees of freedom must be greater than 2 for two-tailed p-value calculation."
    alpha = chi2_dist.pdf(chi2, df)
    mode = df - 2

    if np.isclose(chi2, mode):
        return 1.0

    def root_eq(x):
        return chi2_dist.pdf(x, df) - alpha

    # Find left root
    if chi2 < mode:
        left = chi2_dist.cdf(chi2, df)
    else:
        res_left = root_scalar(root_eq, bracket=[0, mode], method="brentq")
        left = chi2_dist.cdf(res_left.root, df)

    # Find right root
    if chi2 > mode:
        right = chi2_dist.sf(chi2, df)
    else:
        res_right = root_scalar(root_eq, bracket=[mode, 10000 * df], method="brentq")
        right = chi2_dist.sf(res_right.root, df)

    return left + right


def hdp_null_test(key: jax.Array, g: jnp.ndarray, s: jnp.ndarray, T: int, multinomial=False):
    """
    Monte-Carlo exact (probability-ordering) two-sided p-value that the
    PDF samples s (nsamp, nsim) arise from the same distribution as the
    ground truth samples g (nsim,).

    Args:
        key: JAX random key
        g: (log)densities of the ground truth parameters (nsim,)
        s: (log)densities of the sampled parameters (nsamp, nsim)
        T: number of Monte Carlo replications

    See reference:

    ```
    @article{10.1093/mnras/stv1110,
        author = {Harrison, Diana and Sutton, David and Carvalho, Pedro and Hobson, Michael},
        title = {Validation of Bayesian posterior distributions using a multidimensional Kolmogorov–Smirnov test},
        journal = {Monthly Notices of the Royal Astronomical Society},
        volume = {451},
        number = {3},
        pages = {2610-2624},
        year = {2015},
        month = {06},
        issn = {0035-8711},
        doi = {10.1093/mnras/stv1110},
        url = {https://doi.org/10.1093/mnras/stv1110},
        eprint = {https://academic.oup.com/mnras/article-pdf/451/3/2610/4011597/stv1110.pdf},
    }
    ```



    """
    nsamp, nsim = s.shape
    if multinomial:
        return multinomial_null_pvalue(key, jnp.sum(s > g, axis=0), nsamp, T)

    q = jnp.sum(s > g, axis=0)
    p = (1.0 + q) / (nsamp + 1.0)
    chi2 = -2 * jnp.sum(jnp.log(p))  # assuming p ~U(0,1)
    p_value = two_tailed_p(chi2, df=2 * nsim)
    return p_value


# -------- Example --------
if __name__ == "__main__":
    for i in range(20):
        key = random.PRNGKey(i)
        M = 500  # categories {0,...,5}
        N = 100  # sample size
        T = 2000  # Monte Carlo replications

        key, k_obs = random.split(key)
        c = random.randint(k_obs, (N,), minval=0, maxval=M + 1)

        pval = multinomial_null_pvalue(key, c, M, T)
        print("Monte-Carlo exact p-value:", float(pval))
