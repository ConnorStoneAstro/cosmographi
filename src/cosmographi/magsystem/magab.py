from typing import Optional
from .base import MagSystem
import jax.numpy as jnp
from ..throughput.base import Throughput
from ..utils import flux


class MagAB(MagSystem):
    def flux_norm(self, band_i: int, throughput: Throughput):
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
        flux_norm : jnp.ndarray
            Normalization flux for each filter in the AB magnitude system in (electrons/s/cm^2).
        """
        # AB magnitude system normalization flux in erg/s/cm^2/Hz
        f = 3631 * 1e-23  # Convert Jy to erg/s/cm^2/Hz
        nu = flux.nu(throughput.w[band_i])[::-1]
        return flux.f_nu_band(nu, f, throughput.T_nu(nu, band_i))

    def __call__(self, fluxes: jnp.ndarray, flux_norm: Optional[jnp.ndarray] = None):
        """
        Convert fluxes to magnitudes using the AB magnitude formula:

        m_AB = -2.5 * log10(flux / flux_norm)

        Parameters
        ----------
        fluxes : jnp.ndarray
            Fluxes in the magnitude system (flux / flux_norm) for each filter.
        flux_norm : jnp.ndarray, optional
            Normalization flux for each filter. If not provided, it is assumed
            that the input fluxes are already normalized (i.e., flux /
            flux_norm). If provided, the input fluxes will be divided by the
            normalization flux before converting to magnitudes.

        Returns
        -------
        mags : jnp.ndarray
            Magnitudes corresponding to the input fluxes.
        """
        if flux_norm is not None:
            fluxes = fluxes / flux_norm
        # Convert fluxes to magnitudes using the AB magnitude formula
        return -2.5 * jnp.log10(fluxes)

    def err(self, fluxes: jnp.ndarray, flux_errs: jnp.ndarray):
        """
        Convert flux errors to magnitude errors using error propagation for the AB magnitude formula.

        Parameters
        ----------
        fluxes : jnp.ndarray
            Fluxes in the magnitude system (flux / flux_norm) for each filter.
        flux_errs : jnp.ndarray
            Errors on the fluxes in the magnitude system for each filter.

        Returns
        -------
        mag_errs : jnp.ndarray
            Magnitude errors corresponding to the input flux errors.
        """
        # Error propagation for m_AB = -2.5 * log10(fluxes)
        return 2.5 * (flux_errs / fluxes) / jnp.log(10)
