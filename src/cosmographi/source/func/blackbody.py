from ...utils.constants import c_cm, h, k
import jax.numpy as jnp


def blackbody_luminosity_density(w, T, R, N=1):
    """
    Calculate the blackbody luminosity density at a given wavelength.

    Parameters
    ----------
    w : jnp.ndarray
        Wavelength array (in nm).
    T : float
        Temperature of the blackbody (in Kelvin).
    R : float
        Radius of the blackbody emitter (in cm).
    N : float, optional
        Normalization factor, default is 1.

    Returns
    -------
    jnp.ndarray
        Luminosity density array corresponding to wavelength (in erg/s/nm).
    """
    w_cm = w * 1e-7

    # Calculate the specific intensity
    exponent = (h * c_cm) / (w_cm * k * T)
    intensity = jnp.pi * (2 * h * c_cm**2) / (w_cm**5 * (jnp.exp(exponent) - 1))
    luminosity_density = 4 * jnp.pi * R**2 * N * intensity  # in erg/s/cm

    return luminosity_density * 1e-7  # Convert to erg/s/nm
