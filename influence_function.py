import numpy as np
import jax
import jax.numpy as jnp

from jax import grad, jvp, jit
from jax import tree_leaves

from model import loss
from model import predict
from model import one_hot


def hvp(params, x, t, v):
    return jvp(grad(lambda params: loss(params, x, t)), (params,), (v,))[1]


def single_loss(params, sentence, targets):
    preds = predict(params, sentence)
    return -jnp.mean(preds * targets)


@jit
def lissa_estimate(params, x, t, v, h_estimate, damp=0.01, scale=25):
    # Recursively caclulate h_estimate
    hv = hvp(params, x, t, h_estimate)
    h_estimate = jax.tree_multimap(
        lambda x, y, z: x + (1 - damp) * y - z / scale,
        v, h_estimate, hv,
    )
    return h_estimate


def get_s_test(z_test, t_test, params, z_loader, damp=0.01, scale=25.0,
               recursion_depth=5000):
    v = grad(single_loss)(params, z_test, t_test)
    h_estimate = v.copy()
    for depth in range(recursion_depth):
        x, t = next(iter(z_loader))
        t = one_hot(t, n_targets)
        h_estimate = lissa_estimate(params, x, t, v, h_estimate,
                                    damp=damp, scale=scale)

        if depth % 500 == 0:
            print("Calc. s_test recursions: ", depth, recursion_depth)

    return h_estimate


@jit
def get_influence(x, t, params, s_test, N):
    grad_z_vec = grad(single_loss)(params, x, t)
    tmp_influence = jax.tree_multimap(lambda x, y: x * y, grad_z_vec, s_test)
    tmp_influence = -np.sum([jnp.sum(i)
                             for i in tree_leaves(tmp_influence)]) / N
    return tmp_influence
