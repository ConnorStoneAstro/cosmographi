import cosmographi as cg
import jax.numpy as jnp
import jax
from astropy.cosmology import FlatwCDM


def test_flat_lcdm():
    ap_cosmo = FlatwCDM(name="SNLS3+WMAP7_0", H0=70, Om0=0.3, w0=-1)
    cg_cosmo = cg.cosmology.Cosmology(H0=70, Omega_m=0.3, Omega_k=0.0, Omega_r=0.0, w0=-1, wa=0.0)

    z = jnp.linspace(0, 10, 1000)

    assert jnp.allclose(cg_cosmo.H(z), ap_cosmo.H(z), rtol=1e-8)

    assert jnp.allclose(
        jax.vmap(cg_cosmo.comoving_distance)(z), ap_cosmo.comoving_distance(z), rtol=1e-8
    )

    assert jnp.allclose(
        jax.vmap(cg_cosmo.luminosity_distance)(z), ap_cosmo.luminosity_distance(z), rtol=1e-8
    )

    assert jnp.allclose(
        jax.vmap(cg_cosmo.angular_diameter_distance)(z),
        ap_cosmo.angular_diameter_distance(z),
        rtol=1e-8,
    )
