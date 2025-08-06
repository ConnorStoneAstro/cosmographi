import jax.numpy as jnp
from caskade import Module, Param, forward

from ..utils.constants import c


class Cosmology(Module):
    """
    Base class for cosmology modules.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H0 = Param("H0", 67.9, description="Hubble constant at z=0", units="km/s/Mpc")
        self.Omega_m = Param(
            "Omega_m", 0.307, description="Matter density parameter at z=0", units="unitless"
        )
        self.Omega_k = Param(
            "Omega_k", 0.0, description="Curvature density parameter at z=0", units="unitless"
        )
        self.Omega_r = Param(
            "Omega_r", 0.0, description="Radiation density parameter at z=0", units="unitless"
        )
        self.Omega_l = Param(
            "Omega_l",
            lambda p: 1.0 - p.Omega_m.value - p.Omega_k.value - p.Omega_r.value,
            link=(self.Omega_m, self.Omega_k, self.Omega_r),
            description="Dark energy density parameter at z=0",
            units="unitless",
        )
        self.w = Param(
            "w", -1.0, description="Dark energy equation of state parameter", units="unitless"
        )

    @forward
    def H(
        self,
        z,
        H0,
        Omega_m,
        Omega_k,
        Omega_r,
        Omega_l,
        w,
    ):
        """
        Calculate the Hubble parameter at redshift z. Units: km/s/Mpc.
        """

        return (
            H0
            * (
                Omega_m * (1 + z) ** 3
                + Omega_k * (1 + z) ** 2
                + Omega_r * (1 + z) ** 4
                + Omega_l * (1 + z) ** (3 * (1 + w))
            )
            ** 0.5
        )

    @forward
    def comoving_distance(
        self,
        z,
    ):
        """
        Calculate the comoving distance to redshift z. Units: Mpc.
        """
        z_steps = jnp.linspace(0, z, 1000)
        integrand = c / self.H(z_steps)
        return jnp.trapezoid(integrand, z_steps)

    @forward
    def transverse_comoving_distance(
        self,
        z,
        H0,
        Omega_k,
    ):
        """
        Calculate the transverse comoving distance to redshift z. Units: Mpc.
        """
        DC = self.comoving_distance(z)
        DH = c / H0
        return jnp.where(
            Omega_k > 0,
            DH * (1 / Omega_k**0.5) * jnp.sinh(Omega_k**0.5 * DC / DH),
            jnp.where(
                Omega_k < 0,
                DH * (1 / jnp.abs(Omega_k) ** 0.5) * jnp.sin(jnp.abs(Omega_k) ** 0.5 * DC / DH),
                DC,
            ),
        )

    @forward
    def luminosity_distance(self, z):
        """
        Compute the luminosity distance to redshift z. Units: Mpc.
        """
        return self.transverse_comoving_distance(z) * (1 + z)

    @forward
    def angular_diameter_distance(self, z):
        """
        Compute the angular diameter distance to redshift z. Units: Mpc.
        """
        return self.transverse_comoving_distance(z) / (1 + z)

    @forward
    def differential_comoving_volume(self, z):
        """
        Compute the differential comoving volume at redshift z, ie dV/dz. Units: Mpc^3.
        """
        return 4 * jnp.pi * c * self.transverse_comoving_distance(z) ** 2 / self.H(z)
