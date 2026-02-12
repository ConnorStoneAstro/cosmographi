from caskade import Param, forward
from .base import BaseSourceEffect


class WeakLensing(BaseSourceEffect):
    def __init__(self, *args, mu_wl=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_wl = Param(
            "mu_wl",
            mu_wl,
            shape=(),
            valid=(0, None),
            description="Magnification due to weak lensing",
        )

    @forward
    def spectral_flux_density(self, w, *args, mu_wl=None, **kwargs):
        fd = super().spectral_flux_density(w, *args, **kwargs)
        return fd * mu_wl
