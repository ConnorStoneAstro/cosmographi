import os

import jax.numpy as jnp
import jax
from caskade import Param, forward
import numpy as np

from .base import TransientSource
from ..utils import load_salt2_surface, load_salt2_colour_law


class SALT2(TransientSource):

    def __init__(
        self,
        x0=None,
        x1=None,
        c=None,
        M=None,
        CL=None,
        phase_nodes=None,
        wavelength_nodes=None,
        name=None,
    ):
        super().__init__(name)
        self.x0 = Param("x0", x0, shape=(), description="Light curve amplitude")
        self.x1 = Param("x1", x1, shape=(), description="Light curve stretch")
        self.c = Param("c", c, shape=(), description="colour")
        self.M = Param("M", M, description="Phase and wavelength dependent SALT2 model surface")
        self.CL = Param("CL", CL, description="Phase independent SALT2 colour law")
        self.phase_nodes = phase_nodes  # Phase nodes of M
        self.wavelength_nodes = wavelength_nodes  # wavelength nodes of M and CL
        self.phase_sampler = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)

    @forward
    def get_model_basis(self, p, M):
        return tuple(self.phase_sampler(p, self.phase_nodes, M[i]) for i in range(M.shape[0]))

    @forward
    def luminosity_density(self, w, t, t0, x0, x1, c, CL):
        M0, M1 = self.get_model_basis(t - t0)
        f_l = x0 * (M0 + x1 * M1) * jnp.exp(c * CL)
        return jnp.interp(w, self.wavelength_nodes, f_l)

    def load_salt2_model(self, directory):
        phase0, wavelength0, M0 = load_salt2_surface(
            os.path.join(directory, "salt2_template_0.dat")
        )

        phase1, wavelength1, M1 = load_salt2_surface(
            os.path.join(directory, "salt2_template_1.dat")
        )

        assert np.all(phase0 == phase1), "Phase gridding does not match for M0 and M1!"
        assert np.all(
            wavelength0 == wavelength1
        ), "Wavelength gridding does not match for M0 and M1!"
        # fixme convert from spectral flux density to luminosity density (should just be 4 * pi * 10pc)
        self.M = np.stack((M0, M1))
        self.phase_nodes = jnp.array(phase0)
        self.wavelength_nodes = jnp.array(wavelength0)

        wavelength, colour = load_salt2_colour_law(
            os.path.join(directory, "salt2_color_dispersion.dat")
        )
        self.CL = jnp.interp(wavelength0, wavelength, colour)
