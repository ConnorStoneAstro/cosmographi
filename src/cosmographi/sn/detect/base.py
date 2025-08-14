from caskade import Module, forward
import jax
import jax.numpy as jnp


class BaseDetect(Module):
    """
    Base class for detection modules.

    Determine the probability of detecting a supernova based on its light curve (LC).
    This class serves as a template for implementing specific detection functions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @forward
    def logP(self, z, mu):
        """
        Calculate the detection function.
        """
        raise NotImplementedError("Subclasses must implement the logP method.")

    @forward
    def sample(self, key, z, mu):
        """
        Sample from the detection function.
        """
        return jnp.log(jax.random.uniform(key, shape=mu.shape)) < self.logP(z, mu)
