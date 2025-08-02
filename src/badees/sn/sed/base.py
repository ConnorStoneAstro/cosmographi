from caskade import Module, Param, forward


class BaseSNSED(Module):
    """
    Base class for supernova spectral energy distribution (SED) modules.
    Intended to represent common properties of supernova SEDs, such as wavelength and flux.
    This class serves as a foundation for more specific supernova SED classes.
    """

    def __init__(self, sn_type, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type

    @forward
    def sed(self, wavelength):
        """
        Calculate the spectral energy distribution at a given wavelength.
        """
        raise NotImplementedError("Subclasses must implement the sed method.")
