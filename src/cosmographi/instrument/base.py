import jax
from ..filters import Filters
from ..source import BaseSource
from ..utils import flux


class BaseInstrument:
    def __init__(self, filters: tuple[Filters], fov=None):
        self.filters = filters
        self.fov = fov

    def band_flux(self, source: BaseSource, *args, **kwargs):
        fluxes = jax.vmap(
            lambda w, T: flux.f_lambda_band(w, source.spectral_flux_density(w, *args, **kwargs), T)
        )(self.filters.w, self.filters.T)
        return fluxes

    def band_photons(self, source: BaseSource, *args, **kwargs):
        photons = jax.vmap(
            lambda w, T: flux.f_lambda_band_photons(
                w, source.spectral_flux_density(w, *args, **kwargs), T
            )
        )(self.filters.w, self.filters.T)
        return photons

    def mag_AB(self, source: BaseSource, *args, **kwargs):
        mag = jax.vmap(
            lambda nu, w, T: flux.mag_AB(
                nu, flux.f_nu(w, source.spectral_flux_density(w, *args, **kwargs))[::-1], T
            )
        )(self.filters.nu, self.filters.w, self.filters.T_nu)
        return mag
