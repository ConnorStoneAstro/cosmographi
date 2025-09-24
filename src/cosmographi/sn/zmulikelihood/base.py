from caskade import Module, forward
from ...cosmology import Cosmology
from ..rates import CombinedSNRate
from ..detect import BaseDetect
from ... import utils
import jax
import jax.numpy as jnp


class ZMuLikelihood(Module):
    def __init__(
        self,
        cosmology: Cosmology,
        ratecombined: CombinedSNRate,
        detect: list[BaseDetect],
        mean,
        cov,
        name=None,
    ):
        super().__init__(name=name)
        self.cosmology = cosmology
        self.rate = ratecombined
        self.detect = detect
        self.mean = mean
        self.cov = cov

    @forward
    def logP_Xd1_theta(self, z, X, cov_t, t):
        # P(z|Omega)
        lp = self.rate.logPz(z)
        # P(t|z, Omega)
        lp = lp + self.rate.logPt_z(t, z)
        # P(d=1 | X, z, t)
        lp = lp + self.detect[t].log_prob(z, X[1])
        # P(X | z, t, Omega)
        mu = 5 * jnp.log10(self.cosmology.luminosity_distance(z)) - 5
        zmu = jnp.array((z, mu))
        lp = lp + jax.scipy.stats.multivariate_normal.logpdf(X, zmu, cov_t)
        return lp

    @forward
    def logP_Xd1(self, X, cov):
        log_like = []
        for t in range(len(self.detect)):
            ll = utils.log_quad(
                jax.vmap(self.logP_Xd1_theta, in_axes=(0, None, None, None)),
                0,
                2,
                args=(X, cov[t], t),
                n=100,
            )
            log_like.append(ll)
        return jax.nn.logsumexp(jnp.array(log_like))

    @forward
    def logP_d1_zmuobs(self, mu_obs, z_obs, cov):
        return self.logP_Xd1(jnp.array((z_obs, mu_obs)), cov)

    @forward
    def logP_d1_zobs(self, z_obs, cov):
        # int mu_obs
        return utils.log_gauss_rescale_integrate(
            jax.vmap(self.logP_d1_zmuobs, in_axes=(0, None, None)),
            6,
            18,
            mu=5 * jnp.log10(self.cosmology.luminosity_distance(jnp.clip(z_obs, 0.01, 2))) - 5,
            sigma=jnp.sqrt(jnp.max(cov[:, 1, 1])),
            args=(z_obs, cov),
            n=50,
        )

    @forward
    def logP_d1(self, cov):
        # int z_obs
        return utils.log_quad(
            jax.vmap(self.logP_d1_zobs, in_axes=(0, None)),
            -1,
            3,
            args=(cov,),
            n=50,
        )

    @forward
    def _log_likelihood(self, X, cov):
        return self.logP_Xd1(X, cov)

    @forward
    def log_likelihood(self):
        return jax.vmap(self._log_likelihood)(self.mean, self.cov).sum() - len(
            self.cov
        ) * self.logP_d1(self.cov[0])
