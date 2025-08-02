from caskade import Module, Param, forward


class BaseSN(Module):
    """
    Base class for supernova modules.
    Intended to represent common properties of supernovae, such as redshift and peculiar velocity.
    This class serves as a foundation for more specific supernova classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.z = Param("z", 0.0, description="Redshift of the supernova", units="dimensionless")
        self.vp = Param("vp", 0.0, description="Peculiar velocity of the supernova", units="km/s")
