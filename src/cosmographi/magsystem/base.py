from ..throughput.base import Filters


class MagSystem:
    def __init__(self):
        self.name = None  # Should be set by subclasses

    def flux_norm(self, filters: Filters):
        """
        Return the normalization flux for the magnitude system and given filters.
        This is used to convert fluxes to magnitudes.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, fluxes):
        """
        Convert fluxes to magnitudes using the normalization flux.

        Parameters
        ----------
        fluxes : jnp.ndarray
            Fluxes in the magnitude system (flux / flux_norm) for each filter.

        Returns
        -------
        mags : jnp.ndarray
            Magnitudes corresponding to the input fluxes.
        """
        raise NotImplementedError("Subclasses must implement this method.")
