import jax

from .base import BaseDetect
from caskade import forward, Param


class mSigmoidDetect(BaseDetect):

    def __init__(self, threshold: float, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = Param(
            "threshold",
            threshold,
            description="50%% detection threshold on distance modulus axis",
            units="mag",
        )
        self.scale = Param(
            "scale",
            scale,
            description="Scale factor for the detection threshold sigmoid, controls width",
            units="mag",
        )

    @forward
    def log_prob(self, z, m, threshold=None, scale=None):
        return jax.nn.log_sigmoid(-(m - threshold) / scale)


class mzSigmoidDetect(BaseDetect):

    def __init__(self, t_b: float, t_m: float, s_b: float = 1.0, s_m: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.t_b = Param(
            "t_b",
            t_b,
            description="50%% detection threshold on distance modulus axis for z=0",
            units="mag",
        )
        self.t_m = Param(
            "t_m",
            t_m,
            description="slope on distance modulus threshold relative to z",
            units="mag",
        )
        self.s_b = Param(
            "s_b",
            s_b,
            description="Scale factor for the detection threshold sigmoid, controls width at z=0",
            units="mag",
        )
        self.s_m = Param(
            "s_m",
            s_m,
            description="slope on scale factor relative to z",
            units="mag",
        )

    @forward
    def log_prob(self, z, m, t_b=None, t_m=None, s_b=None, s_m=None):
        return jax.nn.log_sigmoid(-(m - (t_b + t_m * z)) / (s_b + s_m * z))
