from caskade import Param, forward

from .base import BaseSourceEffect
from ...utils import fp99_extinction_law, fp99_extinction_law_knots


class HostFitzpatrick99Extinction(BaseSourceEffect):
    """

    Citation
    --------
    @ARTICLE{1999PASP..111...63F,
        author = {{Fitzpatrick}, Edward L.},
            title = "{Correcting for the Effects of Interstellar Extinction}",
        journal = {\pasp},
        keywords = {ISM: DUST, EXTINCTION, Astrophysics},
            year = 1999,
            month = jan,
        volume = {111},
        number = {755},
            pages = {63-75},
            doi = {10.1086/316293},
    archivePrefix = {arXiv},
        eprint = {astro-ph/9809387},
    primaryClass = {astro-ph},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    """

    def __init__(self, *args, A_V=None, R_V=3.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V = Param(
            "A_V", A_V, shape=(), description="Fitzpatrick 1999 host dust extinction law scaling"
        )
        self.R_V = Param(
            "R_V",
            R_V,
            shape=(),
            description="Fitzpatrick 1999 host dust extinction law coefficient",
        )

    @forward
    def luminosity_density(self, w, *args, A_V=None, R_V=None, **kwargs):
        ld = super().luminosity_density(w, *args, **kwargs)
        ext = fp99_extinction_law(w, A_V, R_V)
        return ld * 10 ** (-0.4 * ext)


class HostFitzpatrick99_3p1Extinction(BaseSourceEffect):
    """
    Same as HistFitzpatrick99Extinction except with a fixed R_V = 3.1

    Citation
    --------
    @ARTICLE{1999PASP..111...63F,
        author = {{Fitzpatrick}, Edward L.},
            title = "{Correcting for the Effects of Interstellar Extinction}",
        journal = {\pasp},
        keywords = {ISM: DUST, EXTINCTION, Astrophysics},
            year = 1999,
            month = jan,
        volume = {111},
        number = {755},
            pages = {63-75},
            doi = {10.1086/316293},
    archivePrefix = {arXiv},
        eprint = {astro-ph/9809387},
    primaryClass = {astro-ph},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    """

    def __init__(self, *args, A_V=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V = Param(
            "A_V", A_V, shape=(), description="Fitzpatrick 1999 host dust extinction law scaling"
        )
        self.knots = fp99_extinction_law_knots(3.1)

    @forward
    def luminosity_density(self, w, *args, A_V=None, **kwargs):
        ld = super().luminosity_density(w, *args, **kwargs)
        ext = fp99_extinction_law(w, A_V, 3.1, knots=self.knots)
        return ld * 10 ** (-0.4 * ext)
