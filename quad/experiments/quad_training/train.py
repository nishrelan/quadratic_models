import optax
import haiku as hk
import jax
import jax.numpy as jnp
import logging
from functools import partial
from tqdm import tqdm
import sys

log = logging.getLogger(__name__)

def train(params, opt_state, train_step_fn, acc_fn, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            loss, params, opt_state = train_step_fn(params, opt_state, batch)

        epoch_loss = loss
        accuracy = acc_fn(params, train_loader)

        print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
        print("Train Acc: {}".format(accuracy))
    
    return params, opt_state


def standard_train_step(loss_fn, optimizer):
    
    @jax.jit
    def fn(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state
    
    return fn

def mse_loss(model, type=0):

    @jax.jit
    def loss(params, batch):
        xb, yb = batch
        yb = jax.nn.one_hot(yb, num_classes=2)
        preds = model.apply(params, xb)
        diffs = preds - yb
        loss_vals = jnp.sum(jnp.power(diffs, 2), axis=1)
        return jnp.mean(loss_vals)

    @jax.jit
    def simple_loss(params, batch):
        xb, yb = batch
        preds = model.apply(params, xb)
        return jnp.mean(jnp.power(preds - yb, 2))


    return simple_loss

def make_acc_fn(model):


    def batch_acc(params, batch):
        xb, yb = batch
        preds = model.apply(params, xb)
        matches = jnp.argmax(jnp.abs(preds), axis=-1) == yb
        return jnp.sum(matches)
    
    def simple_batch_acc(params, batch):
        xb, yb = batch
        preds = jnp.ravel(model.apply(params, xb))
        return jnp.sum(jnp.where(preds > 0.5, 1, 0) == yb)
        
    
    def acc(params, loader):
        num_correct = 0
        num_points = 0
        for batch in loader:
            xb, yb = batch
            num_correct += simple_batch_acc(params, batch)
            num_points += len(xb)
        
        return num_correct / num_points
    
    return acc


            


