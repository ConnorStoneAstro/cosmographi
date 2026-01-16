from caskade import Module


class BaseSurvey(Module):
    """
    Base class for representing astronomical surveys.

    The survey is composed of a table of observations, each with attributes such
    as right ascension (RA), declination (Dec), time of observation, and
    observing conditions. Each observation should also be represented by a
    unique key. This way one can query the survey for matching observations to
    get a set of keys, then use those keys to retrieve the relevant observation
    attributes.
    """

    def __init__(self, name=None):
        super().__init__(name)

    def was_observed(self, ra, dec, time):
        raise NotImplementedError("Subclasses must implement this method.")

    def observation_conditions(self, ra, dec, time):
        raise NotImplementedError("Subclasses must implement this method.")

    def match_observations(self, ra, dec, start, end):
        raise NotImplementedError("Subclasses must implement this method.")
