from caskade import Module, forward
import jax.numpy as jnp

from ...cosmology import Cosmology
from ...utils.constants import Mpc_to_cm
from ...utils.bands import bands


class BaseSNSED(Module):
    """
    Base class for supernova spectral energy distribution (SED) modules.
    Intended to represent common properties of supernova SEDs, such as wavelength and flux.
    This class serves as a foundation for more specific supernova SED classes.
    """

    def __init__(self, sn_type, cosmology: Cosmology, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type
        self.cosmology = cosmology

    @forward
    def luminosity_density(self, t, wavelength):
        """
        Calculate the luminosity density at a given wavelength in units of erg/s/nm.
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def flux_density(self, t, wavelength, z):
        """
        Calculate the observed flux density at a given wavelength and z.
        This method should account for z effects on the flux density.

        Here we use:

        $$F_{\\lambda}^{obs}(\lambda_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\lambda}^{rest}(\lambda_{obs} / (1 + z))$$

        where $D_L$ is the luminosity distance in cm. $\\lambda_{obs}$ is the
        observed wavelength. $F_{\\lambda}^{obs}$ is the observed flux density (erg/s/cm^2/nm), and
        $L_{\\lambda}^{rest}$ is the rest-frame luminosity density (erg/s/nm).
        """
        DL = self.cosmology.luminosity_distance(z)  # in Mpc
        DL = DL * Mpc_to_cm  # in cm
        return self.luminosity_density(t, wavelength / (1 + z)) / (
            4 * jnp.pi * (1 + z) * DL**2
        )  # in erg/s/cm^2/nm

    @forward
    def flux_band(self, t, band, z):
        """
        Calculate the observed flux density integrated over a given wavelength band.
        """
        w, T = bands[band]  # w is the wavelengths, T is the transmission function
        Flam = self.flux_density(t, w, z)
        return jnp.trapezoid(Flam * w * T, w) / jnp.trapezoid(w * T, w)  # in erg/s/cm^2/nm
