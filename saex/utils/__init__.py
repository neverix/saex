import importlib
import random
from typing import TypeVar

import equinox as eqx
import jax

T = TypeVar("T")


class RandomNumberGenerator(object):
    def generate(self):
        return get_random_number()

def get_random_number():
    return 5

def get_key():
    return jax.random.PRNGKey(random.randint(0, 2**32 - 1))


def get_obj_from_str(module, cls, reload=False, invalidate_cache=True):
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    module = importlib.import_module(module, package=None)
    obj = module
    for c in cls.split("."):
        obj = getattr(obj, c)
    return module, obj


# copied from https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_stateful.py
def unstatify(model: T) -> tuple[T, eqx.nn.State]:
    # Replace all markers with `int`s. This is needed to ensure that two calls
    # to `make_with_state` produce compatible models and states.
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=eqx.nn._stateful._is_index)
    counter = 0
    new_leaves = []
    for leaf in leaves:
        if eqx.nn._stateful._is_index(leaf):
            leaf = eqx.nn.StateIndex(leaf.init)
            object.__setattr__(leaf, "marker", counter)
            counter += 1
        new_leaves.append(leaf)
    model = jax.tree_util.tree_unflatten(treedef, new_leaves)

    state = eqx.nn.State(model)
    model = eqx.nn._stateful.delete_init_state(model)
    return model, state
