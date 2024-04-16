import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

from . import utils


# implementation of https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition
def toy_model_generator(n_features=2048, n_dimensions=256, corr_temperature=1, frequency_decay=0.99, n_active=5, key=None):
    if key is None:
        key = utils.get_key()
    key, subkey = jax.random.split(key)
    features = jax.random.normal(subkey, (n_features, n_dimensions))
    features = features / jnp.linalg.norm(features, axis=-1, keepdims=True)
    key, subkey = jax.random.split(key)
    correlations = jax.random.normal(subkey, (n_features, n_features)) / (n_features ** 0.5) * corr_temperature
    likelihoods = jnp.power(frequency_decay, jnp.arange(n_features))
    
    def generate_batch(key_gen: jax.random.PRNGKey):
        key_gen, key_probs = jax.random.split(key_gen)
        prob_inputs = jax.random.normal(key_probs, (n_features,))
        prob = jax.scipy.stats.norm.cdf(correlations @ prob_inputs)
        prob = prob * likelihoods
        prob = prob / prob.sum() * n_active
        # checkify.check(prob <= 1.0, "Probabilities must be less than or equal to 1.0")
        prob = prob.clip(0, 1)
        key_gen, key_choice = jax.random.split(key_gen)
        chosen = jax.random.bernoulli(key_choice, prob)
        key_mult = key_gen
        mult = chosen * jax.random.uniform(key_mult, (n_features,))
        batch = (mult[:, None] * features).sum(0)
        return batch, {"mask": 1, "probs": prob, "chosen": chosen, "mult": mult}
    
    return generate_batch, {"features": features, "correlations": correlations, "likelihoods": likelihoods}


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    gen, _ = toy_model_generator(4, 2, n_active=2)
    generations, _ = jax.vmap(gen)(jax.random.split(utils.get_key(), 256))
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.plot([0, 0], [-2, 2], "--", c="black")
    plt.plot([-2, 2], [0, 0], "--", c="black")
    plt.scatter(*np.asarray(generations).T)
    plt.savefig("figures/scratch/toy_model_features.png")
