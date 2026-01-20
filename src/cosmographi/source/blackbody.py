import jax.numpy as jnp
from caskade import Param, forward
from .base import StaticSource, TransientSource
from . import func


class StaticBlackbody(StaticSource):
    """
    Blackbody source module.
    """

    def __init__(self, cosmology, filters, w=None, T=None, R=None, N=1, **kwargs):
        super().__init__(
            cosmology,
            filters,
            w=w,
            LD=lambda p: func.blackbody_luminosity_density(p.w, p.T.value, p.R.value, p.N.value),
            **kwargs,
        )
        self.T = Param(
            "T",
            T,
            description="Blackbody temperature",
            units="K",
        )
        self.R = Param(
            "R",
            R,
            description="Blackbody radius",
            units="cm",
        )
        self.N = Param(
            "N",
            N,
            description="Number of blackbodies with the same temperature and radius",
            units="count",
        )
        self.LD.link((self.T, self.R, self.N))
        self.LD.w = self.w


class TransientBlackbody(TransientSource):
    """
    Blackbody transient source module.
    """

    def __init__(
        self, cosmology, filters, w=None, T=None, R=None, N=1, t0=None, sigma=None, **kwargs
    ):
        super().__init__(cosmology, filters, w=w, **kwargs)
        self.T = Param(
            "T",
            T,
            description="Blackbody temperature",
            units="K",
        )
        self.R = Param(
            "R",
            R,
            description="Blackbody radius",
            units="cm",
        )
        self.N = Param(
            "N",
            N,
            description="Number of blackbodies with the same temperature and radius",
            units="count",
        )
        self.t0 = Param(
            "t0",
            t0,
            description="Time of peak luminosity",
            units="days",
        )
        self.sigma = Param(
            "sigma",
            sigma,
            description="Width of the light curve",
            units="days",
        )

    @forward
    def luminosity_density(self, p, T, R, N, t0, sigma):
        scale = jnp.exp(-0.5 * ((p - t0) / sigma) ** 2)
        return func.blackbody_luminosity_density(self.w, T, R, N) * scale
