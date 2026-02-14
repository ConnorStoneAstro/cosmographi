import jax.numpy as jnp
import jax
from jax import random
from jax.scipy.stats import chi2 as chi2_dist
from scipy.optimize import root_scalar
import numpy as np


def combine_exchangeable_pvalues(pvals):
    """
    Combine p-values using the method of Vovk and Wang (2020).

    Here we use the result for the geometric mean, which is valid for
    any dependence structure.

    ```
    @ARTICLE{2012arXiv1212.4966V,
        author = {{Vovk}, Vladimir and {Wang}, Ruodu},
            title = "{Combining p-values via averaging}",
        journal = {arXiv e-prints},
        keywords = {Mathematics - Statistics Theory, 62G10, 62F03},
            year = 2012,
            month = dec,
            eid = {arXiv:1212.4966},
            pages = {arXiv:1212.4966},
            doi = {10.48550/arXiv.1212.4966},
    archivePrefix = {arXiv},
        eprint = {1212.4966},
    primaryClass = {math.ST},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2012arXiv1212.4966V},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    ```

    See also:
    ```
    @ARTICLE{2025PNAS..12210849G,
        author = {{Gasparin}, Matteo and {Wang}, Ruodu and {Ramdas}, Aaditya},
            title = "{Combining exchangeable P-values}",
        journal = {Proceedings of the National Academy of Science},
        keywords = {Mathematics - Statistics Theory},
            year = 2025,
            month = mar,
        volume = {122},
        number = {11},
            eid = {e2410849122},
            pages = {e2410849122},
            doi = {10.1073/pnas.2410849122},
    archivePrefix = {arXiv},
        eprint = {2404.03484},
    primaryClass = {math.ST},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2025PNAS..12210849G},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    ```
    """
    K = len(pvals)
    lp = jnp.log(pvals)
    GM = jnp.cumsum(lp) / jnp.arange(1.0, K + 1.0)
    return jnp.clip(jnp.exp(1 + jnp.min(GM)), 0, 1)


def dominate_combine_exchangeable_pvalues(pvals):  # fixme not working
    K = len(pvals)
    KM = []
    lp = jnp.log(pvals)
    for l in range(1, K + 1):
        m = jnp.arange(1.0, l + 1.0)
        GM = jnp.cumsum(l * lp[:l]) / m
        KM.append(jnp.clip(jnp.min(jnp.exp(l / m + GM)), 0, 1))
    KM = jnp.array(KM)
    return jnp.clip(jnp.min(KM), 0, 1)


def fisher_combine_independent_pvalues(pvals):
    """
    Combine p-values using Fisher's method, which assumes independence.

    See:
    ```
    @article{fisher1932statistical,
      title={Statistical methods for research workers},
      author={Fisher, R.A.},
      year={1932},
      publisher={Oliver and Boyd}
    }
    ```
    """
    chi2 = -2 * jnp.sum(jnp.log(pvals))
    p_combined = chi2_dist.sf(chi2, df=2 * len(pvals))
    return p_combined


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


def hdp_null_test(g: jnp.ndarray, s: jnp.ndarray):
    """
    exact (probability-ordering) two-sided p-value that the
    posterior samples s (nsamp, nsim) arise from the same distribution as the
    ground truth samples g (nsim,).

    Args:
        g: (log)densities of the ground truth parameters (nsim,)
        s: (log)densities of the sampled parameters (nsamp, nsim)

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
    q = jnp.sum(s >= g[None], axis=0)
    p = jnp.minimum(jnp.ones_like(q), (1.0 + q) / (nsamp + 1.0))
    chi2 = -2 * jnp.sum(jnp.log(p))  # assuming p ~ U(0,1), then -2 log p ~ chi2_2
    p_value = two_tailed_p(chi2, df=2 * nsim)
    return p_value


def ball_null_test(key: jax.Array, g: jnp.ndarray, s: jnp.ndarray, T: int = 1000):
    """
    exact (probability-ordering) two-sided p-value that the
    PDF samples s arise from the same distribution as the
    ground truth samples g.

    This is an extension of TARP to return a p-value.

    Args:
        g: ground truth parameters (nsim, ndim)
        s: sampled parameters (nsamp, nsim, ndim)

    """
    print("WARNING: Still testing ball_null_test implementation.")
    nsamp, nsim, ndim = s.shape
    m = jnp.mean(g, axis=0)
    std = jnp.std(g, axis=0)
    g = (g - m) / std
    s = (s - m) / std

    key, subkey = random.split(key)
    centers = random.normal(subkey, shape=(T, nsim, ndim))  # (T, nsim, ndim)
    r_g = jnp.linalg.norm(g - centers, axis=-1)  # (T, nsim)
    r_s = jnp.linalg.norm(s[:, None] - centers[None], axis=-1)  # (nsamp, T, nsim)
    q = jnp.sum(r_s <= r_g, axis=0)  # (T, nsim)
    p = np.minimum(np.ones_like(q), (1.0 + q) / (nsamp + 1.0))  # (T, nsim)
    print(p)
    p = jnp.median(p, axis=0)  # jax.vmap(combine_exchangeable_pvalues)(p.T)  # (nsim,)
    print(p)
    p = chi2_dist.sf(-2 * jnp.sum(jnp.log(p)), df=2 * nsim)  # ()
    return p


# -------- Example --------
if __name__ == "__main__":
    key = random.PRNGKey(0)
    ndim = 2
    nsim = 100
    nsamp = 1000
    g = random.normal(key, shape=(nsim, ndim))
    s = 2 * random.normal(key, shape=(nsamp, nsim, ndim))
    p = ball_null_test(key, g, s)
    print(p)
