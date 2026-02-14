from ..utils import flux


class Filters:
    """Stores bandpass filters.

    This object holds the wavelength and transmission curves for batched
    filters. The primary attributes are w and T. w is the 1D wavelength (nm)
    array corresponding to the wavelengths at which the transmissions are
    evaluated. T is the transmission array, the first dimension indexes the
    filters, the second gives the transmission corresponding to the wavelength
    array.
    """

    def __init__(self, name, w, T):
        self.name = name
        self.w = w
        if T.ndim == 1:
            T = T[None]
        self.T = T

    @property
    def nu(self):
        return flux.nu(self.w[::-1])

    @property
    def T_nu(self):
        return self.T[:, ::-1]
