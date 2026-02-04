class Filters:
    """Stores bandpass filters.

    This object holds the wavelength and transmission curves for various filters.
    """

    def __init__(self):
        self.filters = {}

    def __getitem__(self, band):
        if band not in self.filters:
            self.get_filter(band)
        return self.filters[band]
