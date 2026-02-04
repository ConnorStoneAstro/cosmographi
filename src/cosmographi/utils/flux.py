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
    """
    return jnp.trapezoid(LD * jnp.where((w >= w1) & (w <= w2), 1, 0), w)  # in erg/s


def f_lambda(z, DL, w, LD):
    """
    Calculate the observed flux density at a given redshift z.
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
        Luminosity distance at redshift z (in Mpc).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    DL = DL * Mpc_to_cm  # in cm
    return w / (1 + z), LD / (4 * jnp.pi * (1 + z) * DL**2)  # in erg/s/cm^2/nm


def f_nu(z, DL, w, LD):
    """
    Calculate the observed flux density at a given redshift z.
    This method should account for z effects on the flux density.

    $$f_{\\nu}^{obs}(\\nu_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\nu}^{rest}(\\nu_{obs} / (1 + z))$$

    where $D_L$ is the luminosity distance in cm. $\\nu_{obs}$ is the
    observed frequency. $f_{\\nu}^{obs}$ is the observed flux density (erg/s/cm^2/Hz), and
    $L_{\\nu}^{rest}$ is the rest-frame luminosity density (erg/s/Hz).

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    w, f_l = f_lambda(z, DL, w, LD)
    nu = c_nm / w  # Hz
    return nu, f_l * c_nm / nu**2


def F_lambda_band(z, DL, w_b, T_b, w, LD):
    """
    Calculate the flux integrated over a given wavelength band.

    $$F_{\\lambda}^{obs} = \\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm).
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    w_z, Flam = f_lambda(z, DL, w, LD)
    return jnp.trapezoid(Flam * jnp.interp(w_z, w_b, T_b), w_z)  # in erg/s/cm^2


def F_nu_band(z, DL, w_b, T_b, w, LD):
    """
    Calculate the observed flux integrated over a given frequency band.

    $$F_{\\nu}^{obs} = \\int f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    nu_z, f_nu = f_nu(z, DL, w, LD)
    nu_b = c_nm / w_b[::-1]
    T_b = T_b[::-1]
    return jnp.trapezoid(f_nu * jnp.interp(nu_z, nu_b, T_b), nu_z)  # in erg/s/cm^2


def f_lambda_band_energy(z, DL, w_b, T_b, w, LD):
    """
    Calculate the observed flux energy density averaged over a given wavelength band.

    $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) T(\\lambda) d\\lambda}{\\int T(\\lambda) d\\lambda}$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm).
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    return F_lambda_band(z, DL, w_b, T_b, w, LD) / jnp.trapezoid(T_b, w_b)  # in erg/s/cm^2/nm


def f_nu_band_energy(z, DL, w_b, T_b, w, LD):
    """
    Calculate the observed flux energy density averaged over a given frequency band.

    $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int T(\\nu) d\\nu}$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    nu_b = c_nm / w_b[::-1]
    T_b = T_b[::-1]
    return F_nu_band(z, DL, w_b, T_b, w, LD) / jnp.trapezoid(T_b, nu_b)  # in erg/s/cm^2/Hz


def f_lambda_band_photons(z, DL, w_b, T_b, w, LD):
    """
    Calculate the observed flux photon density averaged over a given wavelength band.

    $$\\langle f_{\\lambda}^{obs} \\rangle_{band} = \\frac{\\int f_{\\lambda}^{obs}(\\lambda) \\lambda T(\\lambda) d\\lambda}{\\int \\lambda T(\\lambda) d\\lambda}$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    w_z, f_lambda = f_lambda(z, DL, w, LD)  # in erg/s/cm^2/nm
    T_b = jnp.interp(w_z, w_b, T_b)
    return jnp.trapezoid(f_lambda * w_z * T_b, w_z) / jnp.trapezoid(
        T_b * w_z, w_z
    )  # in erg/s/cm^2/nm photon counting


def f_nu_band_photons(z, DL, w_b, T_b, w, LD):
    """
    Calculate the observed flux photon density averaged over a given frequency band.

    $$\\langle f_{\\nu}^{obs} \\rangle_{band} = \\frac{\\int (h\\nu)^{-1}f_{\\nu}^{obs}(\\nu) T(\\nu) d\\nu}{\\int (h\\nu)^{-1}T(\\nu) d\\nu}$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm) observer frame.
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    nu_b = c_nm / w_b[::-1]  # Hz
    T_b = T_b[::-1]
    nu_z, f_nu = f_nu(z, DL, w, LD)  # in erg/s/cm^2/Hz
    T_b = jnp.interp(nu_z, nu_b, T_b)
    return jnp.trapezoid(f_nu * T_b / nu_z, nu_z) / jnp.trapezoid(
        T_b / nu_z, nu_z
    )  # in erg/s/cm^2/Hz photon counting


def mag_AB(z, DL, w_b, T_b, w, LD):
    """
    Calculate the AB magnitude in a given band at redshift z.

    $$m_{AB} = -2.5 \\log_{10} \\left(\\frac{\\langle f_{\\nu}^{obs} \\rangle_{band}}{3631 \\text{ Jy}}\\right)$$

    Parameters
    ----------
    z : float
        Redshift.
    DL : float
        Luminosity distance at redshift z (in Mpc).
    w_b : jnp.ndarray
        Wavelength array for the bandpass (in nm).
    T_b : jnp.ndarray
        Transmission array for the bandpass corresponding to w_b (unitless).
    w : jnp.ndarray
        Wavelength array (in nm) rest frame.
    LD : jnp.ndarray
        Luminosity density array corresponding to w (in erg/s/nm).
    """
    f_nu = f_nu_band_photons(z, DL, w_b, T_b, w, LD)
    return -2.5 * jnp.log10(f_nu) - 48.6  # 48.6 is the zero-point for AB magnitudes in Jy
