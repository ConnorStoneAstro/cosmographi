import jax.numpy as jnp
from .constants import Mpc_to_cm, c_nm


def luminosity(w1, w2, w, LD):
    """
    Calculate the total luminosity in a given wavelength range.
    This method integrates the luminosity density over the specified wavelength range.

    Parameters
    ----------
    w1 : float
        Lower bound of the wavelength range (in nm).
    w2 : float
        Upper bound of the wavelength range (in nm).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).

    Returns
    -------
    luminosity : jnp.ndarray
        integrated between w1 and w2 (erg/s)
    """
    return jnp.trapezoid(LD * jnp.where((w >= w1) & (w <= w2), 1, 0), w)  # in erg/s


def rest_to_observer_wavelength(w_rest, z):
    """Convert rest frame wavelengths to observer frame wavelengths propagating by redshift z"""
    return w_rest * (1 + z)


def observer_to_rest_wavelength(w_obs, z):
    """Convert observer frame wavelengths to rest frame wavelengths propagating by redshift z"""
    return w_obs / (1 + z)


def rest_to_observer_time(t_rest, z):
    return t_rest * (1 + z)


def observer_to_rest_time(t_obs, z):
    return t_obs / (1 + z)


def f_lambda(z, DL, LD):
    """
    Calculate the observed spectral  flux density at a given redshift z.
    This method should account for z effects on the flux density.

    Here we use:

    $$f_{\\lambda}^{obs}(\\lambda_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\lambda}^{rest}(\\lambda_{obs} / (1 + z))$$

    where $D_L$ is the luminosity distance in cm. $\\lambda_{obs}$ is the
    observed wavelength. $f_{\\lambda}^{obs}$ is the observed flux density (erg/s/cm^2/nm), and
    $L_{\\lambda}^{rest}$ is the rest-frame luminosity density (erg/s/nm).

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (Mpc).
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).

    Returns
    -------
    wavelength : jnp.ndarray
        Observe frame wavelength propogation of the input w array (nm)
    spectral_flux_density : jnp.ndarray
        Spectral flux density in (erg/s/cm^2/nm)
    """
    DL = DL * Mpc_to_cm  # in cm
    return LD / (4 * jnp.pi * (1 + z) * DL**2)  # in erg/s/cm^2/nm


def f_nu(w_l, f_l):
    """
    Calculate the observed flux density in frequency units converted from wavelength units.

    $$f_{\\nu}^{obs}(\\nu_{obs}) = \\frac{c}{\\nu^2}f_{\\lambda}^{obs}$$

    and

    $$\\nu = \\frac{c}{\\lambda}$$

    Using the same arguments as `f_lambda` the frequency version would look like:

    $$f_{\\nu}^{obs}(\\nu_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\nu}^{rest}(\\nu_{obs} / (1 + z))$$

    where $D_L$ is the luminosity distance in cm. $\\nu_{obs}$ is the
    observed frequency. $f_{\\nu}^{obs}$ is the observed flux density (erg/s/cm^2/Hz), and
    $L_{\\nu}^{rest}$ is the rest-frame luminosity density (erg/s/Hz).

    Parameters
    ----------
    w_l : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Flux density from f_lambda function (erg/s/cm^2/nm)

    Returns
    -------
    Frequency : jnp.ndarray
        Frequency, nu, array (Hz) observer frame
    spectral_flux_density : jnp.array
        Spectral flux density, f_nu, in frequency space (erg/s/cm^2/Hz)
    """
    nu = c_nm / w_l  # Hz
    return nu, f_l * c_nm / nu**2


def f_lambda_band(w_l, f_l, w_b, T_b):
    """
    Calculate the flux integrated over a given wavelength band.

    $$F_{\\lambda}^{obs} = \\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda$$

    Parameters
    ----------
    w_l : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Flux density from f_lambda function (erg/s/cm^2/nm)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    return jnp.trapezoid(f_l * jnp.interp(w_l, w_b, T_b), w_l)  # in erg/s/cm^2


def f_nu_band(nu, f_nu, w_b, T_b):
    """
    Calculate the observed flux integrated over a given frequency band.

    $$F_{\\nu}^{obs} = \\int f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency space array (Hz) observer frame
    f_nu : jnp.ndarray
        Spectral flux density from f_nu function (erg/s/cm^2/Hz)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    nu_b = c_nm / w_b[::-1]
    T_b = T_b[::-1]
    return jnp.trapezoid(f_nu * jnp.interp(nu, nu_b, T_b), nu)  # in erg/s/cm^2


def f_lambda_band_energy(w_l, f_l, w_b, T_b):
    """
    Calculate the observed flux energy density averaged over a given wavelength band.

    $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda}{\\int T(\\lambda) d\\lambda}$$

    Parameters
    ----------
    w_l : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Flux density from f_lambda function (erg/s/cm^2/nm)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    return f_lambda_band(w_l, f_l, w_b, T_b) / jnp.trapezoid(T_b, w_b)  # in erg/s/cm^2/nm


def f_nu_band_energy(nu, f_nu, w_b, T_b):
    """
    Calculate the observed flux energy density averaged over a given frequency band.

    $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int T(\\nu) d\\nu}$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency space array (Hz) observer frame
    f_nu : jnp.ndarray
        Spectral flux density from f_nu function (erg/s/cm^2/Hz)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    nu_b = c_nm / w_b[::-1]
    T_b = T_b[::-1]
    return f_nu_band(nu, f_nu, w_b, T_b) / jnp.trapezoid(T_b, nu_b)  # in erg/s/cm^2/Hz


def f_lambda_band_photons(w_l, f_l, w_b, T_b):
    """
    Calculate the observed flux photon density averaged over a given wavelength band.

    $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) \\lambda T(\\lambda) d\\lambda}{\\int \\lambda T(\\lambda) d\\lambda}$$

    Parameters
    ----------
    w_l : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Flux density from f_lambda function (erg/s/cm^2/nm)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    T_b = jnp.interp(w_l, w_b, T_b)
    return jnp.trapezoid(f_l * w_l * T_b, w_l) / jnp.trapezoid(
        T_b * w_l, w_l
    )  # in erg/s/cm^2/nm photon counting


def f_nu_band_photons(nu, f_nu, w_b, T_b):
    """
    Calculate the observed flux photon density averaged over a given frequency band.

    $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int (h\\nu)^{-1}f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int (h\\nu)^{-1}T(\\nu) d\\nu}$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency space array (Hz) observer frame
    f_nu : jnp.ndarray
        Spectral flux density from f_nu function (erg/s/cm^2/Hz)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    nu_b = c_nm / w_b[::-1]  # Hz
    T_b = T_b[::-1]
    T_b = jnp.interp(nu, nu_b, T_b)
    return jnp.trapezoid(f_nu * T_b / nu, nu) / jnp.trapezoid(
        T_b / nu, nu
    )  # in erg/s/cm^2/Hz photon counting


def mag_AB(nu, f_nu, w_b, T_b):
    """
    Calculate the AB magnitude in a given band at redshift z.

    $$m_{AB} = -2.5 \\log_{10} \\left(\\frac{\\langle f_{\\nu}^{obs} \\rangle_{band}}{3631 \\text{ Jy}}\\right)$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency space array (Hz) observer frame
    f_nu : jnp.ndarray
        Spectral flux density from f_nu function (erg/s/cm^2/Hz)
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    """
    f_nu = f_nu_band_photons(nu, f_nu, w_b, T_b)
    return -2.5 * jnp.log10(f_nu) - 48.6  # 48.6 is the zero-point for AB magnitudes in Jy
