import pandas as pd
import numpy as np


def load_salt2_surface(surface_file):
    df = pd.read_csv(
        surface_file,
        sep="\s+",
        names=("phase", "wavelength", "fluxdensity"),
    )
    # 1. Get unique values for axes (sorted to ensure consistency)
    # This handles non-regular spacing automatically
    phase_axis = np.sort(df["phase"].unique())
    wavelength_axis = np.sort(df["wavelength"].unique())

    # 2. Create the empty grid (Phase x Wavelength)
    # We initialize with NaN to identify missing data points if any
    grid = np.full((len(phase_axis), len(wavelength_axis)), np.nan)

    # 3. Populate the grid
    # Map the coordinates to indices
    phase_map = {val: i for i, val in enumerate(phase_axis)}
    wave_map = {val: j for j, val in enumerate(wavelength_axis)}

    # Apply the mapping to the dataframe to get index columns
    phase_idx = df["phase"].map(phase_map).values
    wave_idx = df["wavelength"].map(wave_map).values
    flux_vals = df["flux"].values

    # Place values into the grid using advanced indexing
    grid[phase_idx, wave_idx] = flux_vals

    return phase_axis, wavelength_axis, grid


def load_salt2_colour_law(colour_file):
    df = pd.read_csv(colour_file, sep="\s+", names=("wavelength", "dispersion"), comment="#")
    return df["wavelength"].values, df["dispersion"].values
