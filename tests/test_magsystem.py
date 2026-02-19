import cosmographi as cp
import jax.numpy as jnp


def test_MagAB():
    mag_system = cp.MagAB()
    T = cp.RubinThroughput()
    C = cp.Cosmology()
    BB = cp.StaticBlackbody(cosmology=C, z=1.0, T=5000, R=6e10, N=1e9)
    flux_norm = mag_system.flux_norm(0, T)
    f = BB.spectral_flux_density(T.w[0])
    F = cp.utils.flux.f_lambda_band(T.w[0], f, T.T[0])
    mag = mag_system(F, flux_norm)
    assert jnp.isclose(mag, 29.806073915794535), (
        "AB magnitude of T=5000K blackbody should be 29.806073915794535."
    )
