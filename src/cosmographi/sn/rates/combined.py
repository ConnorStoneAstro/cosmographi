from .base import BaseSNRate


class CombinedSNRate(BaseSNRate):
    def __init__(self, sn_rates):
        super().__init__()
        self.sn_rates = sn_rates

    def Ptype(self, z, t):
        return self.sn_rates[t].rate_density(z) / self.rate_density(z)

    def rate_density(self, z):
        return sum(sn_rate.rate_density(z) for sn_rate in self.sn_rates)
