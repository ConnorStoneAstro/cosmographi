import jax.numpy as jnp
from .base import Throughput
from ..utils import bands


class RubinThroughput(Throughput):
    def __init__(self):
        self.bands = ["LSST_u", "LSST_g", "LSST_r", "LSST_i", "LSST_z", "LSST_y"]
        super().__init__(
            self.bands,
            jnp.stack(list(bands[b][0] for b in self.bands)),
            jnp.stack(list(bands[b][1] for b in self.bands)),
        )
