from caskade import Module
import jax


class Sampler(Module):
    def __init__(self, cosmology, rates, source, survey, detection, name=None):
        super().__init__(name=name)
        self.cosmology = cosmology
        self.rates = rates
        self.source = source
        self.survey = survey
        self.detection = detection

    def sample(self, key, n_samples):
        # redshift
        key, subkey = jax.random.split(key)
        z = self.rates.sample_z(subkey, n_samples)

        # Type
        key, subkey = jax.random.split(key)
        obj_type = self.rates.sample_type(subkey, z)

        # Intrinsic light curve parameters
        key, subkey = jax.random.split(key)
        lc_params = self.source.sample_params(subkey, z, obj_type)

        # Sample when SN peak occurs uniformly over survey duration
        key, subkey = jax.random.split(key)
        times = self.survey.sample_times(subkey, n_samples)

        # Sample observation conditions (band, observation time, etc.)
        key, subkey = jax.random.split(key)
        obs_conditions = self.survey.sample_observation_conditions(subkey, times)

        # Compute true fluxes
        key, subkey = jax.random.split(key)
        F = self.source.fluxes(subkey, z, obj_type, lc_params, obs_conditions)

        # Sampler observed fluxes
        key, subkey = jax.random.split(key)
        f = self.survey.observe_fluxes(subkey, F, obs_conditions)

        # Sample selection function to determine detection
        key, subkey = jax.random.split(key)
        d = self.detection.sample_detection(subkey, f)

        return {
            "z": z,
            "obj_type": obj_type,
            "lc_params": lc_params,
            "times": times,
            "obs_conditions": obs_conditions,
            "true_fluxes": F,
            "observed_fluxes": f,
            "detection": d,
        }
