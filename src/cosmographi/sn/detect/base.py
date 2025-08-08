from caskade import Module, forward


class BaseDetect(Module):
    """
    Base class for detection modules.

    Determine the probability of detecting a supernova based on its light curve (LC).
    This class serves as a template for implementing specific detection functions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @forward
    def detection_function(self, LC):
        """
        Calculate the detection function.
        """
        raise NotImplementedError("Subclasses must implement the detection_function method.")
