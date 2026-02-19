from typing import Type
from .base import Source
from .effects.base import SourceEffect


def source_factory(base_source: Type[Source], *effects: tuple[Type[SourceEffect]]) -> Type[Source]:
    """
    Creates a new class that inherits from the effects (mixins) and the base
    source class.

    Organize the effects as though the photons are moving left to right, from
    the "base_source" to the observer on the right.
    """
    assert issubclass(base_source, Source), "base_source should be a Source model"
    assert all(issubclass(eff, SourceEffect) for eff in effects), "Effects should be SourceEffects"

    # Name the class dynamically for better debugging
    name = f"{base_source.__name__}"
    for effect in effects:
        name += f"_{effect.__name__}"

    new_source = type(name, (*reversed(effects), base_source), {})

    return new_source
