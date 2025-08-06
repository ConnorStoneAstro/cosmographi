from caskade import Param, forward
from jax.numpy import interp

from .base import BaseSNRate


class SNRateConst(BaseSNRate):
    """
    Constant supernova rate module.
    """

    def __init__(self, sn_type, rate, **kwargs):
        super().__init__(sn_type, **kwargs)
        self.rate = Param(
            "rate",
            rate,
            description="Supernova rate per unit volume",
            units="1/yr/Mpc^3",
        )

    @forward
    def sn_rate(self, z, rate):
        """
        Calculate the supernova rate at redshift z.
        """
        return rate


class SNRateInterp(BaseSNRate):
    """
    1D linear interpolating supernova rate module.
    """

    def __init__(self, sn_type, rate_z_nodes, rate, **kwargs):
        super().__init__(**kwargs)
        self.sn_type = sn_type
        self.rate_z_nodes = rate_z_nodes
        self.rate = Param(
            "rate",
            rate,
            description="Supernova rate per unit volume at redshift z",
            units="1/yr/Mpc^3",
        )

    @forward
    def sn_rate(self, z, rate):
        """
        Calculate the supernova rate at redshift z.
        """
        # Placeholder for actual calculation
        return interp(z, self.rate_z_nodes, rate)
