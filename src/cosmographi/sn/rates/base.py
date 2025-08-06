from caskade import Module


class BaseSNRate(Module):
    """
    Base class for supernova rate modules.
    """

    def __init__(self, sn_type, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type

    def sn_rate(self, z):
        """
        Calculate the supernova rate at redshift z.
        """
        raise NotImplementedError("Subclasses must implement the sn_rate method.")
