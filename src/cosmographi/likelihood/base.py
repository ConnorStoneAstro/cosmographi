from caskade import Module


class BaseLikelihood(Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
