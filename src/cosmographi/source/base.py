from caskade import Module, forward, Param
import jax.numpy as jnp

from ..cosmology import Cosmology
from ..filters import Filters
from ..utils import flux


class BaseSource(Module):
    """
    Base class for representing astronomical sources.

    This class serves as a foundation for more specific source types, such as stars,
    galaxies, or supernovae. It can be extended to include common properties and methods
    relevant to all source types.
    """

    def __init__(self, cosmology: Cosmology, filters: Filters, name=None):
        super().__init__(name)
        self.cosmology = cosmology
        self.filters = filters


class StaticSource(BaseSource):
    """
    Class for representing static astronomical sources.

    This class is intended for sources that do not vary over time, such as stars or galaxies.
    It can be extended to include properties and methods specific to static sources.
    """

    def __init__(
        self, cosmology: Cosmology, filters: Filters, w: jnp.array, LD: jnp.array, name=None
    ):
        super().__init__(cosmology, filters, name)
        self.w = w  # in nm
        self.LD = Param("LD", LD, description="Luminosity density", units="erg/s/nm")

    @forward
    def mag_AB(self, z, band, LD):
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

    def __init__(self, cosmology: Cosmology, filters: Filters, w: jnp.array, name=None):
        super().__init__(cosmology, filters, name)
        self.w = w  # in nm

    @forward
    def luminosity_density(self, p):
        """
        Calculate the luminosity density at a given wavelength in units of erg/s/nm.
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def mag_AB(self, z, band, p):
        """
        Calculate the AB magnitude in a given band at redshift z.
        """
        w, LD = self.luminosity_density(p)
        DL = self.cosmology.luminosity_distance(z)
        w_b, T_b = self.filters[band]

        return flux.mag_AB(z, DL, w_b, T_b, w, LD)
