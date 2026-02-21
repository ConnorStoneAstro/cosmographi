import jax
import jax.numpy as jnp
from caskade import Module, forward

from ..cosmology import Cosmology
from ..utils import quad


class BaseSNRate(Module):
    """
    Base class for supernova rate modules.
    """

    def __init__(
        self, cosmology: Cosmology, z_min: float, z_max: float, solid_angle=4 * jnp.pi, **kwargs
    ):
        super().__init__(**kwargs)
        self.cosmology = cosmology
        self.z_min = z_min
        self.z_max = z_max
        self.solid_angle = solid_angle

    def rate_density(self, z):
        """
        Calculate the supernova rate density at redshift z.
        """
        raise NotImplementedError("Subclasses must implement the rate_density method.")

    def rate(self, z1, z2):
        """
        Calculate the rate of supernovae per year between redshifts z1 and z2.
        """
        # z = jnp.linspace(z1, z2, 1000)
        # rate_density = self.rate_density(z)
        # vdcv = jax.vmap(self.cosmology.differential_comoving_volume)
        # dVdz = vdcv(z)
        # return jnp.trapezoid(rate_density * dVdz, z)
        integrand = lambda z: self.rate_density(z) * self.cosmology.differential_comoving_volume(z)
        return self.solid_angle * quad(integrand, z1, z2, n=20)

    def expectation(self, z1, z2, t):
        """
        Calculate the expected number of supernovae between redshifts z1 and z2 over time t.
        """
        rate = self.rate(z1, z2)
        return rate * t

    @forward
    def _P(self):
        z = jnp.linspace(self.z_min, self.z_max, 10000)
        rate_density = self.rate_density(z)
        vdcv = jax.vmap(self.cosmology.differential_comoving_volume)
        dVdz = vdcv(z)
        rate = self.solid_angle * rate_density * dVdz
        P = rate / jnp.trapezoid(rate, z)
        return z, P

    @forward
    def logPz(self, z):
        # z_vals, p_vals = self._P()
        # return jnp.log(jnp.interp(z, z_vals, p_vals) + 1e-10)
        rate = lambda z: self.rate_density(z) * self.cosmology.differential_comoving_volume(z)
        return jnp.log(rate(z) / quad(rate, self.z_min, self.z_max, n=20) + 1e-10)

    @forward
    def CDF(self):
        z_vals, p_vals = self._P()
        dz = z_vals[1] - z_vals[0]
        return z_vals, jnp.cumsum(p_vals) * dz

    @forward
    def sample_z(self, key, shape):
        """
        Sample supernovae redshift from the rate distribution.
        """
        z_vals, cdf = self.CDF()
        u = jax.random.uniform(key, shape)
        return jnp.interp(u, cdf, z_vals)
