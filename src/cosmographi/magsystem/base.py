from abc import abstractmethod
from ..throughput.base import Throughput


class MagSystem:
    def __init__(self):
        self.name = None  # Should be set by subclasses

    @abstractmethod
    def flux_norm(self, throughput: Throughput):
        """
        Return the normalization flux for the magnitude system and given throughput.
        This is used to convert fluxes to magnitudes.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
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

    @abstractmethod
    def err(self, fluxes, flux_errs):
        """
        Convert flux errors to magnitude errors using error propagation.

        Parameters
        ----------
        fluxes : jnp.ndarray
            Fluxes in the magnitude system (flux / flux_norm) for each filter.
        flux_errs : jnp.ndarray
            Errors on the fluxes in the magnitude system (flux / flux_norm) for each filter.

        Returns
        -------
        mag_errs : jnp.ndarray
            Magnitude errors corresponding to the input flux errors.
        """
        raise NotImplementedError("Subclasses must implement this method.")
