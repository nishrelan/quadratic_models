import haiku as hk
import jax
import jax.numpy as jnp
from quad.experiments.quad_training.train import train, standard_train_step, make_acc_fn, mse_loss
from quad.data_utils.mnist import get_mnist_binary
from quad.models.mlp import create_model
import sys
import os
import optax


def main():
    project_path = '/Users/nishant/Desktop/Research/kernels/quad/quadratic_models'
    data_path = os.path.join(project_path, 'quad/data/')
    key = jax.random.PRNGKey(0)
    splitter, *keys = jax.random.split(key, num=3)
    train_loader, test_loader = get_mnist_binary(data_path, batch_size=64, train_size=1000, test_size=1000, rngs=keys)
    xb, _ = next(iter(train_loader))
    splitter, key = jax.random.split(splitter)
    model, init_params = create_model(splitter, xb, size=[50000, 1])
    print(len(train_loader))

    optimizer = optax.sgd(1e-4)
    init_opt_state = optimizer.init(init_params)
    loss = mse_loss(model)
    acc_fn = make_acc_fn(model)
    update_fn = standard_train_step(loss, optimizer)

    trained_params, opt_state = train(
        init_params, init_opt_state, update_fn, acc_fn, train_loader, num_epochs=30
    )








if __name__ == '__main__':
    main()