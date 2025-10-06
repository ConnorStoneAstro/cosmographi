from caskade import Module, forward, Param
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
        var_bump=None,
        name=None,
    ):
        super().__init__(name=name)
        self.cosmology = cosmology
        self.rate = ratecombined
        self.detect = detect
        self.mean = mean
        self.cov = cov
        if var_bump is None:
            var_bump = jnp.zeros((len(detect), 2))
        self.var_bump = Param(
            "var_bump",
            var_bump,
            description="Additional variance to add to each SN z and mu, based on type",
            units="unitless,mag^2",
        )

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
    def logP_Xd1(self, X, cov, var_bump=None):
        log_like = []
        for t in range(len(self.detect)):
            ll = utils.log_quad(
                jax.vmap(self.logP_Xd1_theta, in_axes=(0, None, None, None)),
                0,
                2,
                args=(X, cov[t] + jnp.diag(var_bump[t]), t),
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
            sigma=1.2 * jnp.sqrt(jnp.max(cov[:, 1, 1])),
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
    def _log_likelihood(self, i):
        return self.logP_Xd1(self.mean[i], self.cov[i])

    @forward
    def log_likelihood(self):
        ll = jax.vmap(self._log_likelihood)(jnp.arange(len(self.mean))).sum()

        x = self.build_params_array()
        ll_norm = jax.vmap(utils.RBF, in_axes=(None, 0, 0, None, None, None))(
            x, self.reference_points, self.weights, self.scale, utils.gaussian_kernel, -1
        ).sum()

        return ll - ll_norm

    def initialize_logP_d1(self, key, param_bounds, num_points=50):
        key, subkey = jax.random.split(key)
        self.reference_points = utils.superuniform(
            subkey, (len(self.mean), num_points), len(param_bounds), 10, param_bounds.T
        )

        logP_d1_vals = jax.vmap(jax.vmap(self.logP_d1, in_axes=(None, 0)))(
            self.cov, self.reference_points
        )

        self.scale = param_bounds[:, 1] - param_bounds[:, 0]
        self.weights = jax.vmap(lambda *x: utils.RBF_weights(*x, degree=-1), in_axes=(0, 0, None))(
            self.reference_points, logP_d1_vals, self.scale
        )
