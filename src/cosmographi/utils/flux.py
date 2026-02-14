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
    """Convert rest frame time to observer frame time, transformed by redshift z"""
    return t_rest * (1 + z)


def observer_to_rest_time(t_obs, z):
    """Convert observer frame time to rest frame time, transformed by redshift z"""
    return t_obs / (1 + z)


def f_lambda(z, DL, LD):
    """
    Calculate the observed spectral flux density at a given redshift z.
    This method should account for z effects on the spectral flux density.

    Here we use:

    $$f_{\\lambda}^{obs}(\\lambda_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\lambda}^{rest}(\\lambda_{obs} / (1 + z))$$

    where $D_L$ is the luminosity distance in cm. $\\lambda_{obs}$ is the
    observed wavelength. $f_{\\lambda}^{obs}$ is the observed spectral flux density (erg/s/cm^2/nm), and
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


def nu(w):
    """
    Convert from wavelength (nm) to frequency (Hz).

    Parameters
    ----------
    w : jnp.ndarray
        Wavelength array (nm)

    Returns
    -------
    Frequency : jnp.ndarray
        Frequency, nu, array (Hz) same frame as wavelength

    """
    return c_nm / w


def f_nu(w_l, f_l):
    """
    Calculate the observed spectral flux density in frequency units converted from wavelength units.

    $$f_{\\nu}^{obs}(\\nu_{obs}) = \\frac{c}{\\nu^2}f_{\\lambda}^{obs}$$

    and

    $$\\nu = \\frac{c}{\\lambda}$$

    Using the same arguments as `f_lambda` the frequency version would look like:

    $$f_{\\nu}^{obs}(\\nu_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\nu}^{rest}(\\nu_{obs} / (1 + z))$$

    where $D_L$ is the luminosity distance in cm. $\\nu_{obs}$ is the
    observed frequency. $f_{\\nu}^{obs}$ is the observed spectral flux density (erg/s/cm^2/Hz), and
    $L_{\\nu}^{rest}$ is the rest-frame luminosity density (erg/s/Hz).

    Parameters
    ----------
    w_l : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Spectral flux density from f_lambda function (erg/s/cm^2/nm)

    Returns
    -------
    spectral_flux_density : jnp.array
        Spectral flux density, f_nu, in frequency space (erg/s/cm^2/Hz)
    """
    return f_l * w_l**2 / c_nm


def f_l(nu, f_nu):
    """
    Calculate the observed spectral flux density in wavelength units converted from frequency units.

    $$f_{\\lambda}^{obs}(\\lambda_{obs}) = \\frac{\\nu_{obs}^2}{c}f_{\\nu}^{obs}(\\nu_{obs})$$

    and

    $$\\nu = \\frac{c}{\\lambda}$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency array (nm) observer frame
    f_nu : jnp.array
        Spectral flux density from f_nu function (erg/s/cm^2/Hz) observer frame

    Returns
    -------
    spectral_flux_density : jnp.array
        Spectral flux density, f_l, in wavelength space (erg/s/cm^2/nm)
    """
    return f_nu * nu**2 / c_nm


def f_lambda_band(w, f_l, T_b):
    """
    Calculate the flux integrated over a given wavelength band.

    $$F_{\\lambda}^{obs} = \\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda$$

    Parameters
    ----------
    w : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Spectral flux density from f_lambda function (erg/s/cm^2/nm) evaluated at w.
    T_b : jnp.ndarray
        Transmission array for the bandpass (unitless) evaluated at w.
    """
    return jnp.trapezoid(f_l * T_b, w)  # in erg/s/cm^2


def f_nu_band(nu, f_nu, T_nu):
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
    return jnp.trapezoid(f_nu * T_nu, nu)  # in erg/s/cm^2


def f_lambda_band_energy(w, f_l, T_b):
    """
    Calculate the observed flux energy density averaged over a given wavelength band.

    $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda}{\\int T(\\lambda) d\\lambda}$$

    Parameters
    ----------
    w : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Spectral flux density from f_lambda function (erg/s/cm^2/nm) evaluated at w.
    T_b : jnp.ndarray
        Transmission array for the bandpass (unitless) evaluated at w.
    """
    return f_lambda_band(w, f_l, T_b) / jnp.trapezoid(T_b, w)  # in erg/s/cm^2/nm


def f_nu_band_energy(nu, f_nu, T_nu):
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
    return f_nu_band(nu, f_nu, T_nu) / jnp.trapezoid(T_nu, nu)  # in erg/s/cm^2/Hz


def f_lambda_band_photons(w, f_l, T_b):
    """
    Calculate the observed flux photon density averaged over a given wavelength band.

    $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) \\lambda T(\\lambda) d\\lambda}{\\int \\lambda T(\\lambda) d\\lambda}$$

    Parameters
    ----------
    w : jnp.ndarray
        Wavelength array (nm) observer frame
    f_l : jnp.array
        Spectral flux density from f_lambda function (erg/s/cm^2/nm) evaluated at w
    T_b : jnp.ndarray
        Transmission array for the bandpass (unitless) evalauted at w.
    """
    return jnp.trapezoid(f_l * w * T_b, w) / jnp.trapezoid(
        T_b * w, w
    )  # in erg/s/cm^2/nm photon counting


def f_nu_band_photons(nu, f_nu, T_nu):
    """
    Calculate the observed flux photon density averaged over a given frequency band.

    $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int (h\\nu)^{-1}f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int (h\\nu)^{-1}T(\\nu) d\\nu}$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency space array (Hz) observer frame
    f_nu : jnp.ndarray
        Spectral flux density from f_nu function (erg/s/cm^2/Hz) evaluated at nu.
    T_b : jnp.ndarray
        Transmission array for the bandpass (unitless) evaluated at nu.
    """
    return jnp.trapezoid(f_nu * T_nu / nu, nu) / jnp.trapezoid(
        T_nu / nu, nu
    )  # in erg/s/cm^2/Hz photon counting


def mag_AB(nu, f_nu, T_nu):
    """
    Calculate the AB magnitude in a given band at redshift z.

    $$m_{AB} = -2.5 \\log_{10} \\left(\\frac{\\langle f_{\\nu}^{obs} \\rangle_{band}}{3631 \\text{ Jy}}\\right)$$

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency space array (Hz) observer frame
    f_nu : jnp.ndarray
        Spectral flux density from f_nu function (erg/s/cm^2/Hz) evaluated at nu.
    T_b : jnp.ndarray
        Transmission array for the bandpass (unitless) evaluated at nu.
    """
    F_nu = f_nu_band_photons(nu, f_nu, T_nu)
    return -2.5 * jnp.log10(F_nu) - 48.6  # 48.6 is the zero-point for AB magnitudes in Jy
