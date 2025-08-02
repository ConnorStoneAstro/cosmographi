from caskade import Module, Param, forward


class SNRate(Module):
    """
    Base class for supernova rate modules.
    """

    def __init__(self, sn_type, rate_z_nodes, rate, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type
        self.rate_z_nodes = rate_z_nodes
        self.rate = Param("rate", rate, description="Supernova rate at redshift z", units="1/Myr")

    @forward
    def sn_rate(self, z, rate):
        """
        Calculate the supernova rate at redshift z.
        """
        # Placeholder for actual calculation
        return interp1d(z, self.rate_z_nodes, rate)
