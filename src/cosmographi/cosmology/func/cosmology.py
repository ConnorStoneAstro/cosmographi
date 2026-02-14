import jax.numpy as jnp


def mu_to_luminosity_distance(mu):
    """
    Compute the luminosity distance from the distance modulus, which is: 10 ** ((mu + 5) / 5)

    Parameters
    ----------
    mu : jnp.ndarray
        Distance modulus in mag.

    Returns
    -------
    DL : jnp.ndarray
        Luminosity distance in pc.
    """
    return 10 ** ((mu + 5) / 5)


def luminosity_distance_to_mu(DL):
    """
    Compute the distance modulus from the luminosity distance, which is: 5 * log10(luminosity_distance) - 5

    Parameters
    ----------
    DL : jnp.ndarray
        Luminosity distance in pc.

    Returns
    -------
    mu : jnp.ndarray
        Distance modulus in mag.
    """
    return 5 * jnp.log10(DL) - 5
