from .base import BaseSNSED
from caskade import Param, forward
import jax.numpy as jnp

from ...utils.constants import c_cm, h, k


class BlackbodySNSED(BaseSNSED):
    """
    Blackbody supernova spectral energy distribution (SED) module.
    """

    def __init__(self, sn_type, cosmology, T=None, R=None, N=1, **kwargs):
        super().__init__(sn_type, cosmology, **kwargs)
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

    @forward
    def luminosity_density(self, t, wavelength, T=None, R=None, N=None):
        """
        Calculate the luminosity density at a given wavelength in units of erg/s/nm.
        """
        # Convert wavelength from nm to cm
        wavelength_cm = wavelength * 1e-7

        # Calculate the specific intensity
        exponent = (h * c_cm) / (wavelength_cm * k * T)
        intensity = jnp.pi * (2 * h * c_cm**2) / (wavelength_cm**5 * (jnp.exp(exponent) - 1))
        luminosity_density = 4 * jnp.pi * R**2 * N * intensity  # in erg/s/cm

        return luminosity_density * 1e-7  # Convert to erg/s/nm
