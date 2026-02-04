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
    def logP_tzmu(self, t, z, mu):
        # P(z|Omega)
        lp = self.rate.logPz(z)
        # P(t|z, Omega)
        lp = lp + self.rate.logPt_z(t, z)
        # P(mu|z, Omega), currently not used, delta function
        return lp

    @forward
    def logP_X_tzmu(self, X, t, z, mu, cov_t):
        # P(X | t, z, mu, Omega)
        zmu = jnp.array((z, mu))
        lp = jax.scipy.stats.multivariate_normal.logpdf(X, zmu, cov_t)
        return lp

    @forward
    def logP_d1_X(self, X, t):
        # P(d=1 | X, t)
        lp = self.detect[t].log_prob(*X)
        return lp

    @forward
    def logP_Xd1_integrand(self, X, t, z, cov_t):
        # int dmu P(mu|z, Omega) --> distance_modulus(z, Omega)
        mu = self.cosmology.distance_modulus(z)

        lp = self.logP_tzmu(t, z, mu)
        lp = lp + self.logP_X_tzmu(X, t, z, mu, cov_t)
        return lp

    @forward
    def logP_Xd1(self, X, cov, var_bump):
        logP = []
        for t in range(len(self.detect)):
            lp = self.logP_d1_X(X, t)
            lp = lp + utils.log_quad(
                jax.vmap(self.logP_Xd1_integrand, in_axes=(None, None, 0, None)),
                0,
                2,
                args=(X, t, cov[t] + var_bump * jnp.eye(2)),
                n=100,
                argnum=2,
            )
            logP.append(lp)
        return jax.nn.logsumexp(jnp.array(logP))

    @forward
    def _logP_Xd1_join(self, z_obs, mu_obs, cov):
        return self.logP_Xd1(jnp.array((z_obs, mu_obs)), cov)

    @forward
    def logP_Xd1_norm_integrand(self, z_obs, cov):
        mu = self.cosmology.distance_modulus(jnp.clip(z_obs, 0.01, 2))
        return utils.log_gauss_rescale_integrate(
            jax.vmap(self._logP_Xd1_join, in_axes=(None, 0, None)),
            6,
            18,
            mu=mu,
            sigma=1.2 * jnp.sqrt(jnp.max(cov[:, 1, 1])),
            args=(z_obs, cov),
            n=50,
            argnum=1,
        )

    @forward
    def logP_Xd1_norm(self, cov):
        return utils.log_quad(
            jax.vmap(self.logP_Xd1_norm_integrand, in_axes=(0, None)),
            -1,
            3,
            args=(cov,),
            n=50,
        )

    @forward
    def _log_likelihood(self):
        return jax.vmap(self.logP_Xd1)(self.mean, self.cov).sum()

    @forward
    def log_likelihood(self):
        ll = self._log_likelihood()

        x = self.get_values()
        ll_norm = self.logP_Xd1_norm_RBF(x)

        return ll - ll_norm

    @forward
    def logP_Xd1_norm_RBF(self, x):
        """
        Likelihood normalization via pre-computation and interpolation by an RBF.

        Args:
            x: all hyperparameters, (cosmology and phi)
        """
        # ll_norm = jax.vmap(utils.RBF, in_axes=(None, 0, 0, None, None, None))(
        #     x, self.reference_points, self.weights, self.scale, utils.gaussian_kernel, -1
        # ).sum()
        ll_norm = utils.RBF(
            x, self.reference_points, self.weights, self.scale, utils.gaussian_kernel, -1
        )
        return ll_norm

    def initialize_logP_Xd1_norm_RBF(self, key, param_bounds, num_points=50):
        self.reference_points = utils.latin_hypercube(
            m=1,
            n=num_points,
            d=len(param_bounds),
            bounds=param_bounds.T,
            seed=int(jax.random.key_data(key)[0]),
        )[0]

        # logP_d1_vals = (
        #     utils.vmap_chunked1d(
        #         jax.vmap(self.logP_Xd1_norm, in_axes=(None, 0)),
        #         chunk_size=5,
        #         in_axes=(0, None),
        #         prog_bar=True,
        #     )(self.cov, self.reference_points)
        #     .block_until_ready()
        #     .sum(axis=0)
        # )
        logP_d1_vals = jax.vmap(self.logP_Xd1_norm, in_axes=(None, 0))(
            self.cov[0], self.reference_points
        )  # fixme, use full likelihood (variable cov), not just one of them
        logP_d1_vals = logP_d1_vals * len(self.cov)

        self.scale = param_bounds[:, 1] - param_bounds[:, 0]
        # self.weights = utils.vmap_chunked1d(
        #     lambda *x: utils.RBF_weights(*x, scale=self.scale, degree=-1),
        #     chunk_size=25,
        #     prog_bar=False,
        # )(self.reference_points, logP_d1_vals).block_until_ready()
        print(self.reference_points.shape, logP_d1_vals.shape, self.scale.shape)
        self.weights = utils.RBF_weights(
            self.reference_points, logP_d1_vals, scale=self.scale, degree=-1
        )
