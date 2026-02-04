from caskade import Module, forward, Param

from ..cosmology import Cosmology
from ..utils import flux


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
    def mag_AB(self, band, z, LD):
        """
        Calculate the AB magnitude in a given band at redshift z.
        """
        DL = self.cosmology.luminosity_distance(z)
        w_b, T_b = self.filters[band]
        return flux.mag_AB(z, DL, w_b, T_b, self.w, LD)


class TransientSource(BaseSource):
    """
    Class for representing transient astronomical sources.

    This class is intended for sources that vary over time, such as supernovae or variable stars.
    It can be extended to include properties and methods specific to transient sources.
    """

    def __init__(self, cosmology: Cosmology, z=None, t0=None, name=None):
        super().__init__(cosmology, z=z, name=name)

        self.t0 = Param(
            "t0",
            t0,
            description="Light curve reference time (observer frame)",
            units="seconds",
        )  # fixme think about observer frame vs rest frame time factor of (1+z)

    @forward
    def luminosity_density(self, w, t):
        """
        Calculate the luminosity density at a given wavelength in units of
        erg/s/nm and time in units of seconds.
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def mag_AB(self, band, z, p):
        """
        Calculate the AB magnitude in a given band at redshift z.
        """
        w, LD = self.luminosity_density(p)
        DL = self.cosmology.luminosity_distance(z)
        w_b, T_b = self.filters[band]

        return flux.mag_AB(z, DL, w_b, T_b, w, LD)
