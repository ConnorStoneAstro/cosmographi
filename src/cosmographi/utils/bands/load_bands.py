from glob import glob
import pandas as pd
import os


def load_bands():
    """
    Load the bands data from the CSV files in the bands directory.
    Returns a dictionary where keys are band names and values are tuples of (wavelengths, transmission).
    """
    bands = {}

    for file in glob(os.path.join(os.path.split(__file__)[0], "filters/*/*.dat")):
        band_name = file.split("/")[-2] + "_" + file.split("/")[-1].replace(".dat", "")
        df = pd.read_csv(file, sep="\\s+", comment="#")
        wavelengths = df["wavelength"].values
        transmission = df["transmission"].values
        bands[band_name] = (wavelengths, transmission)
    return bands


bands = load_bands()
