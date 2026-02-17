from ..utils import flux


class Throughput:
    """Stores throughput curves

    These combine bandpass filters, atmosphere throughput, mirror/lens and other hardware throughput.

    This object holds the wavelength and transmission curves for batched
    filters including all effects between the atmosphere and the CCD. The primary attributes are w and T. w is the wavelength (nm)
    array corresponding to the wavelengths at which the transmissions are
    evaluated. T is the transmission array, the first dimension indexes the
    filters, the second gives the transmission corresponding to the wavelength
    array.
    """

    def __init__(self, name, w, T):
        self.name = name
        if w.ndim == 1:
            w = w[None]
        if T.ndim == 1:
            T = T[None]
        self.w = w
        self.T = T

        self.nu = flux.nu(self.w[:, ::-1])
        self.T_nu = self.T[:, ::-1]
