from caskade import forward, Param
import jax
from .base import BaseDetect
from ..utils import int_Phi_N


class MuNCDFDetect(BaseDetect):
    """
    Normal Cumulative Distribution Function (CDF) detection function in redshift and distance modulus space.

    This detection function models the probability of detecting a supernova
    as a function of its distance modulus (mu) using a Normal CDF
    function.

    Attributes:
        threshold (Param): The 50% detection threshold on the distance modulus axis.
        scale (Param): Scale factor for the detection threshold probit, controls width.
    """

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
            description="Scale factor for the detection threshold CDF, controls width",
            units="mag",
        )

    @forward
    def log_prob(self, z, mu, threshold=None, scale=None):
        return jax.scipy.stats.norm.logcdf(-(mu - threshold) / scale)

    @forward
    def logZ_norm(self, mean, sigma, threshold=None, scale=None):
        # int_{-inf}^{inf} CDF(-(mu - threshold) / scale) N(mu|m, s^2) dmu
        return int_Phi_N(threshold, scale, mean, sigma)
