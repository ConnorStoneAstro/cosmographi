import jax.numpy as jnp
from interpax import CubicSpline


# Helper function for the UV component (k = E(lambda-V)/E(B-V))
def f99_uv(x_val, R_V):
    # Constants for the UV analytic formula (x >= 3.7)
    # ------------------------------------------------
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    c5 = 5.9

    # R_V dependent coefficients
    c2 = -0.824 + 4.717 / R_V
    c1 = 2.030 - 3.007 * c2

    x2 = x_val**2

    # Drude profile (D term)
    gamma2 = gamma**2
    x02 = x0**2
    d = x2 / ((x2 - x02) ** 2 + x2 * gamma2)

    # Far-UV curvature (F term)
    f = jnp.zeros_like(x_val)
    mask = x_val >= c5
    y = x_val[mask] - c5
    f[mask] = 0.5392 * (y**2) + 0.05644 * (y**3)

    k = c1 + c2 * x_val + c3 * d + c4 * f
    return k


def fp99_extinction_law_knots(R_V):

    # Anchor points (knots) for the Optical/IR Cubic Spline
    # ----------------------------------------------------
    # These knots define the curve for x < 3.7 (lambda > 2700 A)
    # Knot wavelengths (Angstroms): inf, 26500, 12200, 6000, 5470, 4670, 4110, 2700, 2600
    x_knots = jnp.array(
        [
            0.0,
            1.0e4 / 26500.0,
            1.0e4 / 12200.0,
            1.0e4 / 6000.0,
            1.0e4 / 5470.0,
            1.0e4 / 4670.0,
            1.0e4 / 4110.0,
            1.0e4 / 2700.0,
            1.0e4 / 2600.0,
        ]
    )

    # Calculate k values at knots
    # k = E(lambda-V) / E(B-V) = (A_lambda / E(B-V)) - R_V
    k_knots = jnp.zeros_like(x_knots)

    # IR/Optical knots based on R_V polynomials
    rv2 = R_V**2
    k_knots[0] = -R_V  # At infinite wavelength (x=0), A=0 => k = -Rv
    k_knots[1] = 0.26469 * (R_V / 3.1) - R_V
    k_knots[2] = 0.82925 * (R_V / 3.1) - R_V
    k_knots[3] = -0.422809 + 1.00270 * R_V + 2.13572e-04 * rv2 - R_V
    k_knots[4] = -5.13540e-02 + 1.00216 * R_V - 7.35778e-05 * rv2 - R_V
    k_knots[5] = 0.700127 + 1.00184 * R_V - 3.32598e-05 * rv2 - R_V
    k_knots[6] = (
        1.19456 + 1.01707 * R_V - 5.46959e-03 * rv2 + 7.97809e-04 * R_V * rv2 - 4.45636e-05 * rv2**2
    ) - R_V

    # The last two knots (UV anchors) are calculated using the UV formula
    # to ensure the spline connects smoothly to the UV analytic part.
    k_knots[7] = f99_uv(x_knots[7], R_V)
    k_knots[8] = f99_uv(x_knots[8], R_V)
    return x_knots, k_knots


def fp99_extinction_law(w, A_V, R_V, knots=None):
    """
    Calculate dust extinction using the Fitzpatrick (1999) law.

    Parameters
    ----------
    w : array_like
        Wavelengths (nm).
    A_V : float
        The total extinction in the V band (A_V).
    R_V : float, optional
        The ratio of total to selective extinction A_V / E(B-V). Default is 3.1.

    Returns
    -------
    jnp.ndarray
        Extinction A_lambda at the input wavelengths.
    """
    x = 1.0e3 / w

    if knots is None:
        x_knots, k_knots = fp99_extinction_law_knots(R_V)
    else:
        x_knots, k_knots = knots

    # Create Cubic Spline
    # Fitzpatrick 1999 specifies "natural" boundary conditions (2nd deriv = 0 at ends)
    spline = CubicSpline(x_knots, k_knots, bc_type="natural")

    # Calculate Extinction
    # --------------------
    # Initialize output array
    k_vals = jnp.zeros_like(x)

    # Split into UV (analytic) and Optical/IR (spline)
    # The paper/code standard is to use the spline for lambda > 2700 A (x < 3.7)
    # and the analytic formula for lambda <= 2700 A (x >= 3.7).
    uv_mask = x >= 10000.0 / 2700.0
    k_vals = jnp.where(uv_mask, f99_uv(x, R_V), spline(x))

    # Final conversion to extinction A_lambda
    # A_lambda = E(B-V) * (k + R_V)
    # Since A_V = R_V * E(B-V), we have E(B-V) = A_V / R_V
    # A_lambda = (A_V / R_V) * (k + R_V) = A_V * (1 + k / R_V)

    extinction_mag = A_V * (1.0 + k_vals / R_V)

    return extinction_mag


def calzetti00_extinction_law(wave, A_V, R_V=4.05):
    """
    Calculate dust attenuation using the Calzetti (2000) starburst attenuation curve.

    The curve is defined for the range 0.12 to 2.2 microns (1200 - 22000 Angstroms).
    The standard R_V value for this curve is 4.05 +/- 0.80.

    Parameters
    ----------
    wave : array_like
        Wavelengths (nm) rest frame.
    A_V : float
        The total attenuation in the V band (A_V).
    R_V : float, optional
        The ratio of total to selective attenuation A_V / E(B-V).
        The standard value for starburst galaxies is 4.05.

    Returns
    -------
    np.ndarray
        Attenuation A_lambda at the input wavelengths.
    """
    # Convert wavelength to inverse microns (x)
    x = 1.0e3 / wave

    # Calculate k(lambda)
    # k(lambda) = A(lambda) / E(B-V)
    # The curve is typically defined in two regimes: UV and Optical/IR
    # Regime 1: Optical/NIR (0.63 micron <= lambda <= 2.2 micron)
    # 1/2.2 <= x <= 1/0.63  =>  0.45 <= x <= 1.58
    # However, standard implementations often apply this for all x < 1.58 (lambda > 6300 A)
    # Regime 2: UV (0.12 micron <= lambda < 0.63 micron)
    # 1.58 < x <= 8.33
    mask_ir = x < 1.5873  # 1.0 / 0.63
    k_vals = (
        2.659
        * jnp.where(
            mask_ir,
            (-1.857 + 1.040 * x),
            (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3),
        )
        + R_V
    )

    # Calculate Extinction A_lambda
    # A_lambda = k(lambda) * E(B-V)
    # Since A_V = R_V * E(B-V), we have E(B-V) = A_V / R_V
    # Therefore: A_lambda = k(lambda) * (A_V / R_V)

    extinction_mag = k_vals * (A_V / R_V)

    return extinction_mag
