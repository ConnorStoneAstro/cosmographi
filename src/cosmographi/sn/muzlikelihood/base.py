from caskade import Module, forward


class BaseMuZLikelihood(Module):
    """
    Base class for mu-z based likelihood modules.

    Gives the likelihood of a supernova having a given mu-z pair.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @forward
    def likelihood(self, mu, z):
        """
        Calculate the likelihood of a supernova having a given mu-z pair.
        """
        raise NotImplementedError("Subclasses must implement the likelihood method.")
