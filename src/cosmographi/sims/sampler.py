from caskade import Module, forward
import jax


class Sampler(Module):
    def __init__(self, cosmology, rates, source, survey, detection, name=None):
        super().__init__(name=name)
        self.cosmology = cosmology
        self.rates = rates
        self.source = source
        self.survey = survey
        self.detection = detection

    @forward
    def sample(self, key, n_samples):
        # cosmological redshift
        key, subkey = jax.random.split(key)
        zc = self.rates.sample_z(subkey, n_samples)

        # Type
        key, subkey = jax.random.split(key)
        obj_type = self.rates.sample_type(subkey, zc)

        # Sample when SN peak occurs uniformly over survey duration
        key, subkey = jax.random.split(key)
        times = self.survey.sample_times(subkey, n_samples)

        # sample hosts

        # Sample SN positions within survey footprint
        key, subkey = jax.random.split(key)
        positions = self.survey.sample_positions(subkey, n_samples)

        # Set source zc, and obj_type before sampling params

        # Intrinsic light curve parameters
        key, subkey = jax.random.split(key)
        self.source.sample_lightcurve_params(subkey)

        # Sample observation conditions (band, observation time, etc.)
        key, subkey = jax.random.split(key)
        obs_conditions = self.survey.sample_observation_conditions(
            subkey, times, positions
        )  # thinkmore, seems flimsy

        # Sample peculiar velocities
        key, subkey = jax.random.split(key)
        v_pec = self.cosmology.sample_peculiar_velocity(subkey, zc)

        # Compute total redshift (handle v_pec, cosmological dipole, earth motion, etc.)
        z = self.cosmology.total_redshift(zc, times, positions, v_pec)
        # Compute distance modulus
        mu = self.cosmology.distance_modulus(zc)

        # Compute true fluxes
        key, subkey = jax.random.split(key)
        F = self.source.fluxes(subkey, z, mu, obj_type, obs_conditions)  # thinkmore, seems flimsy

        # Sample observed fluxes (add observational noise)
        key, subkey = jax.random.split(key)
        f = self.survey.observe_fluxes(subkey, F, obs_conditions)

        # Sample selection function to determine detection
        key, subkey = jax.random.split(key)
        d = self.detection.sample_detection(subkey, f)

        return {
            "zc": zc,
            "z": z,
            "obj_type": obj_type,
            "times": times,
            "positions": positions,
            "obs_conditions": obs_conditions,
            "v_pec": v_pec,
            "true_fluxes": F,
            "observed_fluxes": f,
            "detection": d,
        }
