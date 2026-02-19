from caskade import Param, forward

from .base import SourceEffect
from ...utils import calzetti00_extinction_law


class HostExtinction_Calzetti00(SourceEffect):
    def __init__(self, *args, A_V_c00h=None, R_V_c00h=4.05, c00h_active=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V_c00h = Param(
            "A_V_c00h",
            A_V_c00h,
            shape=(),
            description="Calzetti 2000 extinction curve attenuation scale",
        )
        self.R_V_c00h = Param(
            "R_V_c00h",
            R_V_c00h,
            shape=(),
            description="Calzetti 2000 extinction curve shape parameter",
        )
        self.c00h_active = c00h_active

    @forward
    def luminosity_density(self, w, *args, A_V_c00h=None, R_V_c00h=None, **kwargs):
        ld = super().luminosity_density(w, *args, **kwargs)
        if not self.c00h_active:
            return ld
        ext = calzetti00_extinction_law(w, A_V_c00h, R_V_c00h)
        return ld * 10 ** (-0.4 * ext)


class MWExtinction_Calzetti00(SourceEffect):
    def __init__(self, *args, A_V_c00mw=None, R_V_c00mw=4.05, c00mw_active=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V_c00mw = Param(
            "A_V_c00mw",
            A_V_c00mw,
            shape=(),
            description="Calzetti 2000 extinction curve attenuation scale",
        )
        self.R_V_c00mw = Param(
            "R_V_c00mw",
            R_V_c00mw,
            shape=(),
            description="Calzetti 2000 extinction curve shape parameter",
        )
        self.c00mw_active = c00mw_active

    @forward
    def spectral_flux_density(self, w, *args, A_V_c00mw=None, R_V_c00mw=None, **kwargs):
        ld = super().spectral_flux_density(w, *args, **kwargs)
        if not self.c00mw_active:
            return ld
        ext = calzetti00_extinction_law(w, A_V_c00mw, R_V_c00mw)
        return ld * 10 ** (-0.4 * ext)
