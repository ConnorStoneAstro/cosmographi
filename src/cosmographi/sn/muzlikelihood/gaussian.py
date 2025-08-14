import jax
import jax.numpy as jnp
from caskade import forward, Param
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
    def cov(self, theta=None, evals=None):
        Q = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
        L = jnp.diag(evals)
        return Q @ L @ Q.T

    @forward
    def log_likelihood(self, mu, z, mean=None):
        """
        Calculate the log-likelihood of a supernova having a given mu-z pair.
        """
        cov = self.cov()
        icov = jnp.linalg.inv(cov)

        diff = jnp.array([mu, z]) - mean
        return -0.5 * (jnp.log(jnp.linalg.det(cov)) + diff.T @ icov @ diff)

    @forward
    def sample(self, key, shape=None, mean=None, cov=None):
        """
        Sample from the Gaussian distribution.
        """
        return jax.random.multivariate_normal(key, mean, cov, shape=shape)
