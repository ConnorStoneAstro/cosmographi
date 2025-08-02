from caskade import Module, Param, forward


class BaseHost(Module):
    """
    Base class for host galaxy modules.
    Intended to represent common properties of host galaxies, such as redshift and peculiar velocity.
    This class serves as a foundation for more specific host galaxy classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.z = Param("z", 0.0, description="Redshift of the host galaxy", units="dimensionless")
        self.vp = Param("vp", 0.0, description="Peculiar velocity of the host galaxy", units="km/s")
