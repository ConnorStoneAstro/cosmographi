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

    def __init__(self, *args, A_V_f99h=None, R_V_f99h=3.1, fixed_R_V_f99h=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V_f99h = Param(
            "A_V_f99h",
            A_V_f99h,
            shape=(),
            description="Fitzpatrick 1999 host dust extinction law scaling",
        )
        if fixed_R_V_f99h:
            self.R_V_f99h = R_V_f99h
            self.knots = fp99_extinction_law_knots(R_V_f99h)
        else:
            self.R_V_f99h = Param(
                "R_V_f99h",
                R_V_f99h,
                shape=(),
                description="Fitzpatrick 1999 host dust extinction law coefficient",
            )
            self.knots = None

    @forward
    def luminosity_density(self, w, *args, A_V_f99h=None, R_V_f99h=None, **kwargs):
        ld = super().luminosity_density(w, *args, **kwargs)
        if R_V_f99h is None:
            R_V_f99h = self.R_V_f99h
        ext = fp99_extinction_law(w, A_V_f99h, R_V_f99h, knots=self.knots)
        return ld * 10 ** (-0.4 * ext)
