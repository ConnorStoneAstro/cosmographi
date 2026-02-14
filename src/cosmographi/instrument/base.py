import jax
from ..filters import Filters
from ..source import BaseSource
from ..utils import flux


class BaseInstrument:
    def __init__(self, filters: tuple[Filters], fov=None):
        self.filters = filters
        self.fov = fov

    def band_flux(self, source: BaseSource, *args, **kwargs):
        f = source.spectral_flux_density(self.filters.w, *args, **kwargs)
        fluxes = jax.vmap(lambda T: flux.f_lambda_band(self.filters.w, f, T))(self.filters.T)
        return fluxes

    def mag_AB(self, source: BaseSource, *args, **kwargs):
        f = source.spectral_flux_density(self.filters.w, *args, **kwargs)
        mag = jax.vmap(
            lambda T: flux.mag_AB(self.filters.nu, flux.f_nu(self.filters.w, f)[::-1], T)
        )(self.filters.T_nu)
        return mag
