import badees as bd
import jax.numpy as jnp

print(bd.__dict__)

C = bd.cosmology.base.Cosmology()

print(C.H(0.0), C.H(1.0))
print(C.luminosity_distance(1.0))
