import jax.numpy as jnp
import jax
from caskade import Module, forward, Param


class SNAbsMagGaussian(Module):
    """
    Module to represent the absolute magnitude of a supernova.
    This class encapsulates the absolute magnitude as a parameter, allowing for easy integration into supernova models.
    """

    def __init__(self, mean_M: float = None, var_M: float = None, **kwargs):
        super().__init__(**kwargs)
        self.mean_M = Param(
            "mean_M", mean_M, description="Mean absolute magnitude of the supernova", units="mag"
        )
        self.var_M = Param(
            "var_M",
            var_M,
            description="Variance of absolute magnitude of the supernova",
            units="mag^2",
        )

    @forward
    def sample(self, key, shape=(), mean_M=None, var_M=None):
        return jax.random.normal(key, shape) * jnp.sqrt(var_M) + mean_M

    @forward
    def logP_M(self, M, mean_M=None, var_M=None):
        return -0.5 * ((M - mean_M) ** 2 / var_M + jnp.log(2 * jnp.pi * var_M))
