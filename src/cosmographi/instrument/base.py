import jax
import jax.numpy as jnp
from ..throughput import Throughput
from ..magsystem import MagSystem, MagAB
from ..source import BaseSource
from ..utils import flux


class BaseInstrument:
    def __init__(
        self, throughput: Throughput, mag_system: MagSystem = MagAB(), Aeff=None, fov=None
    ):
        self.throughput = throughput
        self.mag_system = mag_system
        self.flux_normalization = self.mag_system.flux_norm(self.throughput)
        self.Aeff = Aeff  # Effective aperture size in cm^2
        self.fov = fov

    def _observe_spectrum(self, band_i, source: BaseSource, *args, **kwargs):
        return source.spectral_flux_density(self.throughput.w[band_i], *args, **kwargs)

    def electron_flux(self, band_i, source: BaseSource, *args, **kwargs):
        """
        Return the flux of a source observed through the instrument's throughput.
        The result is provided in electrons/s/cm^2 rather than being normalized by a magnitude system.

        Parameters
        ----------
        band_i : int
            Index of the filter band to use for the flux calculation.
        source : BaseSource
            The source for which to calculate the flux.
        *args, **kwargs
            Additional arguments to pass to the source's spectral_flux_density method. Note that the wavelength argument (w) is provided by the Throughput object and should not be passed in by the user.

        Returns
        -------
        flux : float
            The observed flux of the source through the specified filter, in electrons/s/cm^2.
        """
        f = self._observe_spectrum(band_i, source, *args, **kwargs)
        F = flux.f_lambda_band(self.throughput.w[band_i], f, self.throughput.T[band_i])
        return F

    def flux_var(self, band_i, exp_time, source: BaseSource, *args, **kwargs):
        """
        Return the flux and its measurement variance of a source observed through the instrument's throughput.

        Parameters
        ----------
        band_i : int
            Index of the filter band to use for the flux error calculation.
        exp_time : float
            Exposure time in seconds.
        source : BaseSource
            The source for which to calculate the flux error.
        *args, **kwargs
            Additional arguments to pass to the source's spectral_flux_density method. Note that the wavelength argument (w) is provided by the Throughput object and should not be passed in by the user.

        Returns
        -------
        flux : float
            The observed flux of the source through the specified filter, normalized by the magnitude system's reference flux.
        var : float
            The variance on the observed flux of the source through the specified filter, normalized by the magnitude system's reference flux.
        """
        F = self.electron_flux(band_i, source, *args, **kwargs)
        return (
            F / self.flux_normalization[band_i],  # Aeff and exp_time cancel
            F / self.Aeff / exp_time / self.flux_normalization[band_i] ** 2,
        )

    def flux(self, band_i, source: BaseSource, *args, **kwargs):
        """
        Return the flux of a source observed through the instrument's throughput.
        The result is normalized (flux / flux_ref) in the magnitude system provided.

        Parameters
        ----------
        band_i : int
            Index of the filter band to use for the flux calculation.
        source : BaseSource
            The source for which to calculate the flux.
        *args, **kwargs
            Additional arguments to pass to the source's spectral_flux_density method. Note that the wavelength argument (w) is provided by the Throughput object and should not be passed in by the user.

        Returns
        -------
        flux : float
            The observed flux of the source through the specified filter, normalized by the magnitude system's reference flux.
        """
        F = self.electron_flux(band_i, source, *args, **kwargs)
        return F / self.flux_normalization[band_i]

    def mag(self, source: BaseSource, *args, **kwargs):
        F = self.flux(source, *args, **kwargs)
        mags = self.mag_system(F)
        return mags

    def observe(self, key, band_i, exp_time, source: BaseSource, *args, **kwargs):
        """
        Create a mock observation of a source through the instrument's
        throughput, including noise.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for generating noise in the observation.
        band_i : int
            Index of the filter band to use for the observation.
        exp_time : float
            Exposure time in seconds.
        source : BaseSource
            The source to observe.
        *args, **kwargs
            Additional arguments to pass to the source's spectral_flux_density method. Note that the wavelength argument (w) is provided by the Throughput object and should not be passed in by the user.

        Returns
        -------
        flux_obs : float
            The observed flux of the source through the specified filter, normalized by the magnitude system's reference flux, including noise.
        flux_err_obs : float
            The observed uncertainty on the flux of the source through the specified filter, normalized by the magnitude system's reference flux, including noise.
        flux_true : float
            The true flux of the source through the specified filter, normalized by the magnitude system's reference flux, without noise.
        flux_err_true : float
            The true uncertainty on the flux of the source through the specified filter, normalized by the magnitude system's reference flux, without noise.

        Note
        ----
        The noise reported by this function is the **measured** noise, meaning
        the observed flux is converted to a number of electrons and the square
        root of the observed number of electrons is used as the measured noise.
        The actual noise generating process comes from the square root on the
        true number of expected electrons. If this is not clear, see the code
        with comments that explain the exact process.
        """
        # True flux, and true flux uncertainty in the magnitude system (flux / flux_ref)
        flux, flux_var = self.flux_var(band_i, exp_time, source, *args, **kwargs)
        # Scale factor between flux in magnitude system and number of electrons
        scale = exp_time * self.Aeff * self.flux_normalization[band_i]

        N = flux * scale  # Expected number of electrons
        Ne = jnp.sqrt(jnp.abs(flux_var)) * scale  # Flux error in electrons

        noise = jax.random.normal(key, shape=flux.shape)
        Nobs = N + noise * Ne  # Observed number of electrons with noise

        flux_obs = Nobs / scale  # Convert back to flux units
        flux_err_obs = jnp.sqrt(jnp.abs(Nobs)) / scale  # Measured flux uncertainty
        return flux_obs, flux_err_obs, flux, jnp.sqrt(flux_var)
