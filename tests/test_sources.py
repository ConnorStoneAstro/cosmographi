import cosmographi as cp
import jax
import jax.numpy as jnp
import sncosmo
import numpy as np

import pytest


@pytest.mark.parametrize(
    "x0, x1, c, z",
    [
        (1e-4, 0.5, 0.1, 1e-9),
        (1e-5, -0.5, -0.1, 0.5),
        (1e-3, 0.0, 0.0, 1.0),
    ],
)
def test_salt2_2021(x0, x1, c, z):

    # Test that the SALT2_2021 source produces a light curve consistent with sncosmo's SALT2 model
    C = cp.Cosmology()
    S = cp.SALT2_2021(cosmology=C, z=z, x0=x0, x1=x1, c=c, t0=0.0)
    S.CL.to_static()
    S.M.to_static()
    S.load_salt2_model()

    # Get the luminosity density from sncosmo's SALT2 model
    source = sncosmo.get_source("salt2", version="T21")
    model = sncosmo.Model(source=source)
    model.set(z=z, t0=0.0, x0=x0, x1=x1, c=c)
    w = jnp.linspace(
        model.minwave() * 1.1, model.maxwave() * 0.9, 500
    )  # Wavelengths in Angstroms (observer frame)
    t = jnp.linspace(
        model.mintime() * 0.9, model.maxtime() * 0.9, 25
    )  # Time in days (observer frame)
    ld_sncosmo = (
        np.stack(list(model.flux(t_, w) for t_ in np.array(t))) * 10
    )  # convert /Angstrom to /nm

    # Get the spectral flux density from our SALT2_2021 implementation
    ld_cp = jax.vmap(S.spectral_flux_density, in_axes=(None, 0))(w, t)

    print(np.array(ld_cp) / ld_sncosmo)
    assert np.allclose(np.array(ld_cp) / ld_sncosmo, 1.0, rtol=1e-3), (
        "Spectral Flux Density from SALT2_2021 should match sncosmo's SALT2 model."
    )
