import importlib
import random
from typing import TypeVar
from unittest.mock import patch

import inspect
import ast
import dis

import equinox as eqx
import jax

T = TypeVar("T")


class ModulePatcher(object):
    def __init__(self, parent, module):
        self._parent = parent
        self._module = module
        self._patches = []
    
    def in_module(self, module):
        return self._parent.in_module(module)
    
    def patch(self, target, new):
        full_name = f"{self._module}.{target}"
        def patch_fn():
            orig = get_obj_from_str(full_name)
            source = inspect.getsource(orig)
            obj_ast = ast.parse(source)
            obj_ast = ast.fix_missing_locations(obj_ast)
            code = compile(obj_ast, filename=orig.__code__.co_filename, mode="exec")
            env = orig.__globals__.copy()
            exec(code, env)
            replacement = env[orig.__name__]
            return replacement
        
        self._patches.append(patch(full_name, new_callable=patch_fn))
        return self

class Patcher(object):
    def __init__(self):
        self._module_patchers = []
    
    def in_module(self, module):
        patcher = ModulePatcher(self, module)
        self._module_patchers.append(patcher)
        return patcher

    def __enter__(self):
        for module_patcher in self._module_patchers:
            for patcher in module_patcher._patches:
                patcher.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for module_patcher in self._module_patchers:
            for patcher in module_patcher._patches:
                patcher.stop()


def get_key():
    return jax.random.PRNGKey(random.randint(0, 2**32 - 1))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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
