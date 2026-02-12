from caskade import Module, forward, Param

from ..cosmology import Cosmology
from ..utils import flux
from ..utils.constants import c_nm


class BaseSource(Module):
    """
    Base class for representing astronomical sources.

    This class serves as a foundation for more specific source types, such as stars,
    galaxies, or supernovae. It can be extended to include common properties and methods
    relevant to all source types.
    """

    def __init__(self, cosmology: Cosmology, z=None, name=None):
        super().__init__(name)
        self.cosmology = cosmology
        self.z = Param("z", z, description="Redshift", units="dimensionless")

    def luminosity_density(self, w):
        raise NotImplementedError("Please use a subclass of BaseSource")

    def spectral_flux_density(self, w):
        raise NotImplementedError("Please use a subclass of BaseSource")


class StaticSource(BaseSource):
    """
    Class for representing static astronomical sources.

    This class is intended for sources that do not vary over time, such as stars or galaxies.
    It can be extended to include properties and methods specific to static sources.
    """

    @forward
    def luminosity_density(self, w):
        """
        Calculate the luminosity density at a given wavelength in units of erg/s/nm.
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def spectral_flux_density(self, w, z):
        ld = self.luminosity_density(w)
        DL = self.cosmology.luminosity_distance(z)
        return flux.f_lambda(z, DL, w, ld)

    @forward
    def spectral_flux_density_frequency(self, nu):
        w = c_nm / nu
        f_l = self.spectral_flux_density(w)
        return flux.f_nu(w, f_l)[1]


class TransientSource(BaseSource):
    """
    Class for representing transient astronomical sources.

    This class is intended for sources that vary over time, such as supernovae or variable stars.
    It can be extended to include properties and methods specific to transient sources.
    """

    def __init__(self, cosmology: Cosmology, z=None, t0=None, p_range=(0, 1), name=None):
        super().__init__(cosmology, z=z, name=name)

        self.t0 = Param(
            "t0",
            t0,
            description="Light curve reference time (observer frame)",
            units="days",
        )
        self.p_range = p_range  # Time (phase) range for transient relative to t0, rest frame

    @forward
    def visible(self, t, t0, z):
        t_range = (
            flux.rest_to_observer_time(self.p_range[0], z),
            flux.rest_to_observer_time(self.p_range[1], z),
        )
        return (t >= t0 + t_range[0]) & (t <= t0 + t_range[1])

    @forward
    def luminosity_density(self, w, p):
        """
        Calculate the luminosity density at a given wavelength in units of
        erg/s/nm and time in units of seconds.

        Parameters
        ----------
        w : jnp.ndarray
            Wavelength array (nm) rest frame
        p : jnp.ndarray
            Time (phase) of observation (days) rest frame
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def spectral_flux_density(self, w, t, z):
        """
        Calculate the observed spectral flux density at a given redshift z.
        This method should account for z effects on the spectral flux density.

        Here we use:

        $$f_{\\lambda}^{obs}(\\lambda_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\lambda}^{rest}(\\lambda_{obs} / (1 + z))$$

        where $D_L$ is the luminosity distance in cm. $\\lambda_{obs}$ is the
        observed wavelength. $f_{\\lambda}^{obs}$ is the observed spectral flux density (erg/s/cm^2/nm), and
        $L_{\\lambda}^{rest}$ is the rest-frame luminosity density (erg/s/nm).

        Parameters
        ----------
        w : jnp.ndarray
            Wavelength array (nm) observer frame.
        t : jnp.ndarray or float
            Time of observation (days) observer frame

        Returns
        -------
        spectral_flux_density : jnp.ndarray
            Spectral flux density in (erg/s/cm^2/nm)
        """
        w = flux.observer_to_rest_wavelength(w, z)
        p = flux.observer_to_rest_time(t, z)
        ld = self.luminosity_density(w, p)
        DL = self.cosmology.luminosity_distance(z)
        return flux.f_lambda(z, DL, ld)

    @forward
    def spectral_flux_density_frequency(self, nu, t):
        w = c_nm / nu
        f_l = self.spectral_flux_density(w, t)
        return flux.f_nu(w, f_l)
