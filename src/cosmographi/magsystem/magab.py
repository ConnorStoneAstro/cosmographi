from .base import MagSystem
import jax.numpy as jnp
import jax
from ..throughput.base import Throughput
from ..utils import flux


class MagAB(MagSystem):
    def __init__(self):
        super().__init__()
        self.name = "AB"

    def flux_norm(self, throughput: Throughput):
        """
        Return the normalization flux for the AB magnitude system and given
        throughput. This is computed assuming a flat spectrum in f_nu
        (erg/s/cm^2/Hz) of 3631 Jy, which is the standard reference flux for AB
        magnitudes.

        Parameters
        ----------
        throughput : Throughput
            The throughput for which to compute the normalization flux.

        Returns
        -------
        norm_flux : jnp.ndarray
            Normalization flux for each filter in the AB magnitude system in (electrons/s/cm^2).
        """
        # AB magnitude system normalization flux in erg/s/cm^2/Hz
        f = 3631 * 1e-23  # Convert Jy to erg/s/cm^2/Hz
        return jax.vmap(flux.f_nu_band, in_axes=(None, None, 0))(throughput.nu, f, throughput.T_nu)

    def __call__(self, fluxes):
        """
        Convert fluxes to magnitudes using the AB magnitude formula:
        m_AB = -2.5 * log10(fluxes / norm_flux)

        Parameters
        ----------
        fluxes : jnp.ndarray
            Fluxes in the magnitude system (flux / flux_norm) for each filter.

        Returns
        -------
        mags : jnp.ndarray
            Magnitudes corresponding to the input fluxes.
        """
        # Convert fluxes to magnitudes using the AB magnitude formula
        return -2.5 * jnp.log10(fluxes)
