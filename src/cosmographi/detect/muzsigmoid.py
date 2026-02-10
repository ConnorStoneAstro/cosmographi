import jax
import jax.numpy as jnp

from .base import BaseDetect
from .. import utils
from caskade import forward, Param


class MuSigmoidDetect(BaseDetect):
    def __init__(self, threshold: float, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = Param(
            "threshold",
            threshold,
            description="50%% detection threshold on distance modulus axis",
            units="mag",
        )
        self.scale = Param(
            "scale",
            scale,
            description="Scale factor for the detection threshold sigmoid, controls width",
            units="mag",
        )

    @forward
    def log_prob(self, z, mu, threshold=None, scale=None):
        return jax.nn.log_sigmoid(-(mu - threshold) / scale)

    @forward
    def logZ_norm(self, mean, sigma, threshold=None, scale=None):
        s = jnp.maximum(sigma, scale)
        mu = utils.midpoints(
            jnp.minimum(mean, threshold) - 5 * s, jnp.maximum(mean, threshold) + 5 * s, 50
        )
        dmu = mu[1] - mu[0]
        pg = jax.scipy.stats.norm.logpdf(mu, mean, sigma)
        ps = self.log_prob(0, mu)
        return jax.nn.logsumexp(pg + ps) + jnp.log(dmu)


class MuZSigmoidDetect(BaseDetect):
    def __init__(self, t_b: float, t_m: float, s_b: float = 1.0, s_m: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.t_b = Param(
            "t_b",
            t_b,
            description="50%% detection threshold on distance modulus axis for z=0",
            units="mag",
        )
        self.t_m = Param(
            "t_m",
            t_m,
            description="slope on distance modulus threshold relative to z",
            units="mag",
        )
        self.s_b = Param(
            "s_b",
            s_b,
            description="Scale factor for the detection threshold sigmoid, controls width at z=0",
            units="mag",
        )
        self.s_m = Param(
            "s_m",
            s_m,
            description="slope on scale factor relative to z",
            units="mag",
        )

    @forward
    def log_prob(self, z, mu, t_b=None, t_m=None, s_b=None, s_m=None):
        return jax.nn.log_sigmoid(-(mu - (t_b + t_m * z)) / (s_b + s_m * z))
