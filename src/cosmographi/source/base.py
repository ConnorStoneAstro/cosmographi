from caskade import Module, forward
import jax.numpy as jnp

from ..cosmology import Cosmology
from ..utils.constants import Mpc_to_cm, c_nm
from ..utils.bands import bands


class BaseSource(Module):
    """
    Base class for representing astronomical sources.

    This class serves as a foundation for more specific source types, such as stars,
    galaxies, or supernovae. It can be extended to include common properties and methods
    relevant to all source types.
    """

    def __init__(self, name=None):
        super().__init__(name)


class StaticSource(BaseSource):
    """
    Class for representing static astronomical sources.

    This class is intended for sources that do not vary over time, such as stars or galaxies.
    It can be extended to include properties and methods specific to static sources.
    """

    pass


class TransientSource(BaseSource):
    """
    Class for representing transient astronomical sources.

    This class is intended for sources that vary over time, such as supernovae or variable stars.
    It can be extended to include properties and methods specific to transient sources.
    """


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
    def luminosity(self, t, w1, w2):
        """
        Calculate the total luminosity in a given wavelength range.
        This method integrates the luminosity density over the specified wavelength range.
        """
        w = jnp.linspace(w1, w2, 1000)
        LD = self.luminosity_density(t, w)
        return jnp.trapezoid(LD, w)  # in erg/s

    @forward
    def f_lambda(self, t, wavelength, z):
        """
        Calculate the observed flux density at a given wavelength and z.
        This method should account for z effects on the flux density.

        Here we use:

        $$f_{\\lambda}^{obs}(\\lambda_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\lambda}^{rest}(\\lambda_{obs} / (1 + z))$$

        where $D_L$ is the luminosity distance in cm. $\\lambda_{obs}$ is the
        observed wavelength. $f_{\\lambda}^{obs}$ is the observed flux density (erg/s/cm^2/nm), and
        $L_{\\lambda}^{rest}$ is the rest-frame luminosity density (erg/s/nm).
        """
        DL = self.cosmology.luminosity_distance(z)  # in Mpc
        DL = DL * Mpc_to_cm  # in cm
        return self.luminosity_density(t, wavelength / (1 + z)) / (
            4 * jnp.pi * (1 + z) * DL**2
        )  # in erg/s/cm^2/nm

    @forward
    def f_nu(self, t, nu, z):
        """
        Calculate the observed flux density at a given frequency and z.
        This method should account for z effects on the flux density.

        $$f_{\\nu}^{obs}(\\nu_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\nu}^{rest}(\\nu_{obs} / (1 + z))$$

        where $D_L$ is the luminosity distance in cm. $\\nu_{obs}$ is the
        observed frequency. $f_{\\nu}^{obs}$ is the observed flux density (erg/s/cm^2/Hz), and
        $L_{\\nu}^{rest}$ is the rest-frame luminosity density (erg/s/Hz).
        """
        return self.f_lambda(t, c_nm / nu, z) * c_nm / nu**2

    @forward
    def F_lambda_band(self, t, band, z):
        """
        Calculate the observed flux integrated over a given wavelength band.

        $$F_{\\lambda}^{obs} = \\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda$$
        """
        w, T = bands[band]
        Flam = self.f_lambda(t, w, z)
        return jnp.trapezoid(Flam * T, w)  # in erg/s/cm^2

    @forward
    def F_nu_band(self, t, band, z):
        """
        Calculate the observed flux integrated over a given frequency band.

        $$F_{\\nu}^{obs} = \\int f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu$$
        """
        w, T = bands[band]  # w is the wavelengths, T is the transmission function
        nu = c_nm / w[::-1]
        T = T[::-1]
        f_nu = self.f_nu(t, nu, z)
        return jnp.trapezoid(f_nu * T, nu)  # in erg/s/cm^2

    @forward
    def f_lambda_band_energy(self, t, band, z):
        """
        Calculate the observed flux energy density averaged over a given wavelength band.

        $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda}{\\int T(\\lambda) d\\lambda}$$
        """
        w, T = bands[band]  # w is the wavelengths, T is the transmission function
        return self.F_lambda_band(t, band, z) / jnp.trapezoid(T, w)  # in erg/s/cm^2/nm

    @forward
    def f_nu_band_energy(self, t, band, z):
        """
        Calculate the observed flux energy density averaged over a given frequency band.

        $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int T(\\nu) d\\nu}$$
        """
        w, T = bands[band]  # w is the wavelengths, T is the transmission function
        nu = c_nm / w[::-1]
        T = T[::-1]
        return self.F_nu_band(t, band, z) / jnp.trapezoid(T, nu)  # in erg/s/cm^2/Hz

    @forward
    def f_lambda_band_photons(self, t, band, z):
        """
        Calculate the observed flux photon density averaged over a given wavelength band.

        $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) \\lambda T(\\lambda) d\\lambda}{\\int \\lambda T(\\lambda) d\\lambda}$$
        """
        w, T = bands[band]  # w is the wavelengths, T is the transmission function
        f_lambda = self.f_lambda(t, w, z)  # in erg/s/cm^2/nm
        return jnp.trapezoid(f_lambda * w * T, w) / jnp.trapezoid(
            T * w, w
        )  # in erg/s/cm^2/nm photon counting

    @forward
    def f_nu_band_photons(self, t, band, z):
        """
        Calculate the observed flux photon density averaged over a given frequency band.

        $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int (h\\nu)^{-1}f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int (h\\nu)^{-1}T(\\nu) d\\nu}$$
        """
        w, T = bands[band]  # w is the wavelengths, T is the transmission function
        nu = c_nm / w[::-1]  # Hz
        T = T[::-1]
        f_nu = self.f_nu(t, nu, z)  # in erg/s/cm^2/Hz
        return jnp.trapezoid(f_nu * T / nu, nu) / jnp.trapezoid(
            T / nu, nu
        )  # in erg/s/cm^2/Hz photon counting

    @forward
    def mag_AB(self, t, band, z):
        """
        Calculate the AB magnitude in a given band at redshift z.

        $$m_{AB} = -2.5 \\log_{10} \\left(\\frac{\\langle f_{\\nu}^{obs} \\rangle_{band}}{3631 \\text{ Jy}}\\right)$$
        """
        f_nu = self.f_nu_band_photons(t, band, z)
        return -2.5 * jnp.log10(f_nu) - 48.6  # 48.6 is the zero-point for AB magnitudes in Jy
