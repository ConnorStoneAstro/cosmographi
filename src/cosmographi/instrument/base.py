from caskade import Module
from ..filters import Filters


class BaseInstrument(Module):
    def __init__(self, filters: Filters, name=None):
        super().__init__(name)
        self.filters = filters
