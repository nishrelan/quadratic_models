import haiku as hk
import jax
from functools import partial

# Define network
class MLP:
    def __init__(self, output_sizes):
        self.mlp = hk.nets.MLP(output_sizes=output_sizes, activation=jax.nn.relu,
                            with_bias=True)

    def __call__(self, x):
        return self.mlp(x)

def mlp_forward(x, size):
    model = MLP(size)
    return model(x)

def create_model(rng_key, sample_data, size=None):
    if size is None:
        size = [300, 100, 10]
    model = hk.without_apply_rng(hk.transform(partial(mlp_forward, size=size)))
    init_params = model.init(rng_key, sample_data)
    return model, init_params