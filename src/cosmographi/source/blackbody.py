import jax.numpy as jnp
from caskade import Param, forward
from .base import StaticSource, TransientSource
from . import func


class StaticBlackbody(StaticSource):
    """
    Blackbody source module.

    This class represents a blackbody source that doesn't vary over time. It is
    characterized by its temperature, radius, and a scale factor N which
    corresponds to the number of blackbodies at the temperature/radius values.
    """

    def __init__(self, cosmology, filters, z=None, w=None, T=None, R=None, N=1, **kwargs):
        super().__init__(
            cosmology,
            filters,
            z=z,
            **kwargs,
        )
        self.T = Param("T", T, shape=(), description="Blackbody temperature", units="K")
        self.R = Param("R", R, shape=(), description="Blackbody radius", units="cm")
        self.N = Param(
            "N",
            N,
            shape=(),
            description="Number of blackbodies with the same temperature and radius",
            units="count",
        )

    @forward
    def luminosity_density(self, w, T, R, N):
        return func.blackbody_luminosity_density(w, T, R, N)


class TransientBlackbody(TransientSource):
    """
    Blackbody transient source module.
    """

    def __init__(self, cosmology, filters, z=None, T=None, R=None, N=1, sigma=None, **kwargs):
        super().__init__(cosmology, filters, z=z, **kwargs)
        self.T = Param("T", T, shape=(), description="Blackbody temperature", units="K")
        self.R = Param("R", R, shape=(), description="Blackbody radius", units="cm")
        self.N = Param(
            "N",
            N,
            shape=(),
            description="Number of blackbodies with the same temperature and radius",
            units="count",
        )

        self.sigma = Param(
            "sigma", sigma, shape=(), description="Width of the light curve", units="days"
        )

    @forward
    def luminosity_density(self, w, t, T, R, N, t0, sigma):
        scale = jnp.exp(-0.5 * ((t - t0) / sigma) ** 2)
        return func.blackbody_luminosity_density(w, T, R, N) * scale
