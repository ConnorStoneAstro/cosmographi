import jax
import jax.numpy as jnp
from caskade import Module, forward, active_cache

from ...cosmology import Cosmology


class BaseSNRate(Module):
    """
    Base class for supernova rate modules.
    """

    def __init__(self, sn_type, cosmology: Cosmology, z_max, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type
        self.cosmology = cosmology
        self.z_max = z_max

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

    @active_cache
    @forward
    def _P(self):
        z = jnp.linspace(0, self.z_max, 1000)
        rate_density = self.rate_density(z)
        vdcv = jax.vmap(self.cosmology.differential_comoving_volume)
        dVdz = vdcv(z)
        dz = z[1] - z[0]
        rate = rate_density * dVdz
        P = rate / jnp.trapezoid(rate, dz=dz)
        return z, P

    @forward
    def P(self, z):
        z_vals, p_vals = self._P()
        return jnp.interp(z, z_vals, p_vals)

    @forward
    def CDF(self):
        z_vals, p_vals = self._P()
        dz = z_vals[1] - z_vals[0]
        return z_vals, jnp.cumsum(p_vals) * dz
