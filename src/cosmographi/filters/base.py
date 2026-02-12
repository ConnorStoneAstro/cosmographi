from ..utils import flux


class Filters:
    """Stores bandpass filters.

    This object holds the wavelength and transmission curves for batched
    filters. The primary attributes are w and T. w is the wavelength (nm) array
    where the first dimension indexes the filters and the second gives the
    wavelengths. T is the transmission array, the first dimension indexes the
    filters, the second gives the transmission corresponding to the wavelength
    array.
    """

    def __init__(self, name, w, T):
        self.name = name
        if w.ndim == 1 and T.ndim == 1:
            w = w[None]
            T = T[None]
        self.w = w
        self.T = T

    @property
    def nu(self):
        return flux.nu(self.w[:, ::-1])

    @property
    def T_nu(self):
        return self.T[:, ::-1]
