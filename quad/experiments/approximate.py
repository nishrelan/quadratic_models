# Hessian-vector products!

from quad.models.mlp import create_model
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import haiku as hk

def model_output(params, model, x):
    return model.apply(params, x)[0][0]


def ravel_pytree(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    return jnp.concatenate([jnp.ravel(elt) for elt in leaves],axis=0)

def tree_dot(a, b):
    sums = jax.tree_util.tree_map(lambda x, y: jnp.sum(x*y), a, b)
    return jnp.sum(jnp.array(jax.tree_util.tree_leaves(sums)))

# forward-over-reverse
def hvp(f, w, v):
    return jax.jvp(jax.grad(f), (w,), (v,))[1]

def get_linear_approx(model, w0, x):
    def linear(w):
        const = model_output(w0, model, x)
        diff = jax.tree_util.tree_map(lambda x,y: x - y, w, w0)
        
        grad = jax.grad(model_output, argnums=0)(w0, model, x)
        

        return const + tree_dot(diff, grad)
    return linear

def get_quad_approx(model, w0, x):

    def quad(w):
        const = model_output(w0, model, x)
        diff = jax.tree_util.tree_map(lambda x,y: x - y, w, w0)
        
        grad = jax.grad(model_output, argnums=0)(w0, model, x)

        f = partial(model_output, model=model, x=x)
        quad = hvp(f, w0, diff)

        return const + tree_dot(diff, grad) + 0.5*tree_dot(diff, quad)

    return quad






        



def main():
    rng = hk.PRNGSequence(jax.random.PRNGKey(2))
    x = jnp.ones((1,28*28))
    model, params = create_model(next(rng), x, size=[300, 100, 1])
    

    lin = get_linear_approx(model, params, x)
    quad = get_quad_approx(model, params, x)
    f = partial(model_output, model=model, x=x)
    

    _, w = create_model(next(rng), x, size=[300,100,1])
    print(lin(w))
    print(quad(w))
    
    alphas = jnp.linspace(0, 0.1, 20)
    f_vals = []
    lins = []
    quads = []
    for alpha in alphas:
        offset = jax.tree_util.tree_map(lambda a, b: a + alpha*b, params, w)
        f_vals.append(f(offset))
        lins.append(lin(offset))
        quads.append(quad(offset))
    
    plt.plot(alphas, f_vals, label='f')
    plt.plot(alphas, lins, label='linear')
    plt.plot(alphas, quads, label='quad')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
