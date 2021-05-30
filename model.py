import jax.numpy as jnp

from jax import random
from jax import jit, grad, vmap
from jax.scipy.special import logsumexp


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


@jit
def update(i, params, x, y, opt_state):
    grads = grad(loss)(params, x, y)
    opt_state = opt_update(i, grads, opt_state)
    return get_params(opt_state), opt_state


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return (scale * random.normal(w_key, (n, m)),
            scale * random.normal(b_key, (n,)))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
