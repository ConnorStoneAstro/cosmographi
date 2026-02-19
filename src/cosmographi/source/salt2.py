import os

import jax.numpy as jnp
import jax
from caskade import Param, forward
import numpy as np

from .base import TransientSource
from ..utils import load_salt2_surface, load_salt2_colour_law, flux
from ..utils.constants import Mpc_to_cm


class SALT2_2021(TransientSource):
    def __init__(
        self,
        cosmology=None,
        x0=None,
        x1=None,
        c=None,
        M=None,
        CL=None,
        phase_nodes=None,
        wavelength_nodes=None,
        name=None,
        **kwargs,
    ):
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.x0 = Param("x0", x0, shape=(), description="Light curve amplitude")
        self.x1 = Param("x1", x1, shape=(), description="Light curve stretch")
        self.c = Param("c", c, shape=(), description="colour")
        self.M = Param(
            "M",
            M,
            shape=(2, None, None),
            dynamic=False,
            description="Phase and wavelength dependent SALT2 model surface M0 and M1 components (luminosity density)",
        )
        self.CL = Param(
            "CL", CL, shape=(None,), dynamic=False, description="Phase independent SALT2 colour law"
        )

        self.phase_nodes = phase_nodes  # Phase nodes of M
        self.wavelength_nodes = wavelength_nodes  # wavelength nodes of M and CL
        self.phase_sampler = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=-1)

    def min_phase(self):
        return self.phase_nodes[0]

    def max_phase(self):
        return self.phase_nodes[-1]

    @forward
    def get_model_basis(self, p, M):
        return tuple(self.phase_sampler(p, self.phase_nodes, M[i]) for i in range(M.shape[0]))

    @forward
    def luminosity_density(self, w, p, t0, x0, x1, c, CL, z):
        """
        Calculate the luminosity density at a given wavelength in units of
        erg/s/nm and time in units of seconds.

        Parameters
        ----------
        w : jnp.ndarray
            Wavelength array (nm) rest frame
        p : jnp.ndarray
            Time (phase) of observation (days) rest frame
        """
        p0 = flux.observer_to_rest_time(t0, z)
        M0, M1 = self.get_model_basis(p - p0)
        f_l = x0 * (M0 + x1 * M1) * jnp.exp(c * CL)
        return jnp.interp(w, self.wavelength_nodes, f_l)

    def load_salt2_model(self, directory=None):
        if directory is None:
            directory = os.path.join(os.path.dirname(__file__), "data/SALT2_2021/")
        phase0, wavelength0, M0 = load_salt2_surface(
            os.path.join(directory, "salt2_template_0.dat")
        )

        phase1, wavelength1, M1 = load_salt2_surface(
            os.path.join(directory, "salt2_template_1.dat")
        )

        assert np.all(phase0 == phase1), "Phase gridding does not match for M0 and M1!"
        assert np.all(wavelength0 == wavelength1), (
            "Wavelength gridding does not match for M0 and M1!"
        )
        # convert from spectral flux density to luminosity density (should just be 4 * pi * 10pc^2)
        self.M = np.stack((M0, M1)) * (4 * np.pi * (10 * 1e-6 * Mpc_to_cm) ** 2)
        self.phase_nodes = jnp.array(phase0)
        # Convert from angstroms to nm
        self.wavelength_nodes = jnp.array(wavelength0) / 10

        wavelength, colour = load_salt2_colour_law(
            os.path.join(directory, "salt2_color_dispersion.dat")
        )
        self.CL = jnp.interp(wavelength0, wavelength, colour)
