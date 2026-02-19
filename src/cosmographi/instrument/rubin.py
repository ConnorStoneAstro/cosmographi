import jax.numpy as jnp
from .base import Instrument
from ..throughput import Throughput, RubinThroughput
from ..magsystem import MagSystem, MagAB


class RubinObservatory(Instrument):
    def __init__(
        self,
        throughput: Throughput = RubinThroughput(),
        mag_system: MagSystem = MagAB(),
        name=None,
        **kwargs,
    ):
        Aeff = jnp.pi * (649 / 2) ** 2  # Effective area of the Rubin Observatory in cm^2
        super().__init__(throughput, Aeff=Aeff, mag_system=mag_system, name=name, **kwargs)
