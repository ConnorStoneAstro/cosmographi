from .base import BaseSource
from .effects.base import BaseSourceEffect


def source_factory(base_source: BaseSource, *effects: tuple[BaseSourceEffect]):
    """
    Creates a new class that inherits from the effects (mixins) and the base
    source class.

    Organize the effects as though the photons are moving left to right, from
    the "base_source" to the observer on the right.
    """
    assert isinstance(base_source, BaseSource), "base_source should be a Source model"
    assert all(isinstance(eff, BaseSourceEffect) for eff in effects), (
        "Effects should be SourceEffects"
    )

    # Name the class dynamically for better debugging
    name = f"{base_source.__name__}"
    for effect in effects:
        name += f"_{effect.__name__}"

    new_source = type(name, (*reversed(effects), base_source), {})

    return new_source
