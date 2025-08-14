import jax

from .base import BaseDetect
from caskade import forward, Param


class MuSigmoid(BaseDetect):

    def __init__(self, threshold: float, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = Param(
            "threshold",
            threshold,
            description="50%% detection threshold on distance modulus axis",
            unit="mag",
        )
        self.scale = Param(
            "scale",
            scale,
            description="Scale factor for the detection threshold sigmoid, controls width",
            unit="mag",
        )

    @forward
    def logP(self, mu, threshold, scale):
        return jax.nn.log_sigmoid(-(mu - threshold) / scale)


class MuZSigmoid(BaseDetect):

    def __init__(self, t_b: float, t_m: float, s_b: float = 1.0, s_m: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.t_b = Param(
            "t_b",
            t_b,
            description="50%% detection threshold on distance modulus axis for z=0",
            unit="mag",
        )
        self.t_m = Param(
            "t_m",
            t_m,
            description="slope on distance modulus threshold relative to z",
            unit="mag",
        )
        self.s_b = Param(
            "s_b",
            s_b,
            description="Scale factor for the detection threshold sigmoid, controls width at z=0",
            unit="mag",
        )
        self.s_m = Param(
            "s_m",
            s_m,
            description="slope on scale factor relative to z",
            unit="mag",
        )

    @forward
    def logP(self, mu, z, t_b, t_m, s_b, s_m):
        return jax.nn.log_sigmoid(-(mu - (t_b + t_m * z)) / (s_b + s_m * z))
