import jax
import jax.numpy as jnp
from caskade import Module

from ...cosmology import Cosmology


class BaseSNRate(Module):
    """
    Base class for supernova rate modules.
    """

    def __init__(self, sn_type, cosmology: Cosmology, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type
        self.cosmology = cosmology

    def rate_density(self, z):
        """
        Calculate the supernova rate density at redshift z.
        """
        raise NotImplementedError("Subclasses must implement the rate_density method.")

    def rate(self, z1, z2):
        """
        Calculate the rate of supernovae per year between redshifts z1 and z2.
        """
        z = jnp.linspace(z1, z2, 1000)
        rate_density = self.rate_density(z)
        vdcv = jax.vmap(self.cosmology.differential_comoving_volume)
        dVdz = vdcv(z)
        return jnp.trapezoid(rate_density * dVdz, z)

    def expectation(self, z1, z2, t):
        """
        Calculate the expected number of supernovae between redshifts z1 and z2 over time t.
        """
        rate = self.rate(z1, z2)
        return rate * t
