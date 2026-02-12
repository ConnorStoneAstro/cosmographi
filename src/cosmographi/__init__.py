__version__ = "0.0.0"
from . import sn, detect, rates, source, survey, likelihood, sims, utils, filters, instrument
from .cosmology import Cosmology

import jax

jax.config.update("jax_enable_x64", True)
import caskade as ck

ck.backend.backend = "jax"

__all__ = (
    "sn",
    "detect",
    "rates",
    "source",
    "survey",
    "likelihood",
    "sims",
    "utils",
    "filters",
    "instrument",
    "Cosmology",
)
