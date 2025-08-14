from typing import List
import jax
import jax.numpy as jnp
from .base import BaseSNRate
from caskade import forward


class CombinedSNRate(BaseSNRate):
    def __init__(self, sn_rates: List[BaseSNRate], **kwargs):
        super().__init__(**kwargs)
        self.sn_rates = sn_rates

    @forward
    def Ptype(self, z, t):
        return self.sn_rates[t].rate_density(z) / self.rate_density(z)

    @forward
    def sample_type(self, key, z):
        rd_t = jnp.array(list(map(lambda sn_rate: sn_rate.rate_density(z), self.sn_rates)))
        return jax.random.choice(key, len(self.sn_rates), p=rd_t)

    @forward
    def rate_density(self, z):
        return sum(sn_rate.rate_density(z) for sn_rate in self.sn_rates)
