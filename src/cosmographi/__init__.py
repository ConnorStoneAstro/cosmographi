__version__ = "0.0.0"
from . import cosmology, sn, utils
from .cosmology import Cosmology

import jax

jax.config.update("jax_enable_x64", True)
import caskade as ck

ck.backend.backend = "jax"
