from .base import BaseSNSED
from caskade import Param, forward
import jax.numpy as jnp

from ...utils.constants import c, h, k


class BlackbodySNSED(BaseSNSED):
    """
    Blackbody supernova spectral energy distribution (SED) module.
    """

    def __init__(self, sn_type, cosmology, T=None, R=None, **kwargs):
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

    @forward
    def luminosity_density(self, t, wavelength, T=None, R=None):
        """
        Calculate the luminosity density at a given wavelength in units of erg/s/nm.
        """
        # Convert wavelength from Angstrom to cm
        wavelength_cm = wavelength * 1e-7
        c_cm = c * 1e2  # speed of light from m/s to cm/s

        # Calculate the specific intensity
        exponent = (h * c_cm) / (wavelength_cm * k * T)
        intensity = (2 * h * c_cm**2) / (wavelength_cm**5 * (jnp.exp(exponent) - 1))

        luminosity_density = 4 * jnp.pi * R**2 * intensity  # in erg/s/cm

        return luminosity_density * 1e7  # Convert to erg/s/nm
