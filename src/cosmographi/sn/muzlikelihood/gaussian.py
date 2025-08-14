import jax.numpy as jnp
from caskade import Module, forward, Param
from .base import BaseMuZLikelihood


class GaussianMuZLikelihood(BaseMuZLikelihood):
    def __init__(
        self,
        mean,
        theta,
        evals,
    ):
        super().__init__()
        self.mean = Param("mean", mean, shape=(2,))
        self.theta = Param("theta", theta, shape=())
        self.evals = Param("evals", evals, shape=(2,))

    @forward
    def log_likelihood(self, mu, z, mean, theta, evals):
        """
        Calculate the log-likelihood of a supernova having a given mu-z pair.
        """
        Q = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
        L = jnp.diag(evals)
        cov = Q @ L @ Q.T
        icov = jnp.linalg.inv(cov)

        diff = jnp.array([mu, z]) - mean
        return -0.5 * (jnp.log(jnp.linalg.det(cov)) + diff.T @ icov @ diff)
