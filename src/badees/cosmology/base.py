from caskade import Module, Param, forward


class Cosmology(Module):
    """
    Base class for cosmology modules.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H0 = Param("H0", 70.0, description="Hubble constant at z=0 in km/s/Mpc")
        self.omega_m = Param("omega_m", 0.3, description="Matter density parameter at z=0")
        self.omega_l = Param("omega_l", 0.7, description="Dark energy density parameter at z=0")

    @forward
    def H(self, z, H0, omega_m, omega_l):
        """
        Calculate the Hubble parameter at redshift z.
        """
        return H0 * (omega_m * (1 + z) ** 3 + omega_l) ** 0.5
