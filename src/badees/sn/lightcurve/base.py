class BaseLightCurve:
    """
    Base class for light curve modules.
    Intended to represent common properties of light curves, such as time and flux.
    This class serves as a foundation for more specific light curve classes.
    """

    def __init__(self, time, wavelength, flux, flux_uncertainty, **kwargs):
        super().__init__(**kwargs)
        self.time = time
        self.wavelength = wavelength
        self.flux = flux
        self.flux_uncertainty = flux_uncertainty

    def likelihood(self, model_flux):
        """
        Calculate the likelihood of the observed flux given a model flux.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the likelihood method.")
