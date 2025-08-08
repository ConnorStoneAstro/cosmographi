import jax.numpy as jnp

from .base import BaseDetect


class ThresholdDetect(BaseDetect):
    """
    Threshold-based detection of supernovae from a light curve (LC). The light
    curve should have shape (n_timesteps, n_bands) and hold values with units of
    flux. If threshold is a scalar, it will be applied to all bands; if it is a
    vector it must match the number of bands.
    """

    def __init__(self, threshold: jnp.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def detect(self, LC: jnp.ndarray) -> bool:
        return jnp.sum(LC > self.threshold) >= 2
