from caskade import Param, forward
from jax.numpy import interp

from .base import BaseSNRate


class SNRateConst(BaseSNRate):
    """
    Constant supernova rate module.
    """

    def __init__(self, sn_type, cosmology, r, **kwargs):
        super().__init__(sn_type, cosmology, **kwargs)
        self.r = Param(
            "r",
            r,
            description="Supernova rate per unit volume",
            units="1/yr/Mpc^3",
        )

    @forward
    def rate_density(self, z, r):
        """
        Calculate the supernova rate at redshift z.
        """
        return r


class SNRateInterp(BaseSNRate):
    """
    1D linear interpolating supernova rate module.
    """

    def __init__(self, sn_type, cosmology, rate_z_nodes, r, **kwargs):
        super().__init__(sn_type, cosmology, **kwargs)
        self.sn_type = sn_type
        self.rate_z_nodes = rate_z_nodes
        self.r = Param(
            "r",
            r,
            description="Supernova rate per unit volume at redshift z",
            units="1/yr/Mpc^3",
        )

    @forward
    def rate_density(self, z, r):
        """
        Calculate the supernova rate at redshift z.
        """
        # Placeholder for actual calculation
        return interp(z, self.rate_z_nodes, r)
