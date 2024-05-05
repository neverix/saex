import ast
import inspect
import sys
from copy import copy
from unittest.mock import patch

from . import get_obj_from_str


class Patcher(object):
    def __init__(self):
        self._patches = []
        self._patcher_traces = {}
        self._module = None
    
    def in_module(self, module):
        self._module = module
        return self

    def patch(self, target, replacements=[], additional_context={}):
        if self._module is None:
            raise ValueError("Must specify module before patching: Patcher().in_module(\"module\")")
        module_name = self._module
        full_name = f"{module_name}.{target}"

        module, orig = get_obj_from_str(module_name, target)
        
        own_stacks = 0
        for callerframerecord in inspect.stack():
            frame = callerframerecord[0]
            info = inspect.getframeinfo(frame)
            if own_stacks == 2:
                break
            if info.filename == __file__:
                own_stacks += 1
        pos_keys = ["lineno", "col_offset", "end_lineno", "end_col_offset"]
        kwargs = {k: getattr(info.positions, k) for k in pos_keys}
        
        source, offset = inspect.getsourcelines(orig)
        indent = len(source[0]) - len(source[0].lstrip())
        source = "".join([line[indent:] for line in source])
        
        filename = inspect.getsourcefile(orig)
        
        orig_ast = ast.parse(source)
        obj_ast = orig_ast
        for source_line, target_line in replacements:
            obj_ast = NodeReplacer(
                ast.parse(source_line).body[0], ast.parse(target_line).body[0],
                offset - 1, indent, filename, self._patcher_traces, kwargs).visit(obj_ast)
        obj_ast.body[0].end_lineno = (obj_ast.body[0].end_lineno - obj_ast.body[0].lineno) + offset + 1
        obj_ast.body[0].lineno = offset + 1
        obj_ast = ast.fix_missing_locations(obj_ast)
        code = compile(obj_ast, filename=filename, mode="exec")
        # env = {**orig.__globals__, **additional_context}
        env = {**module.__dict__, **additional_context}
        exec(code, env)
        replacement = env[orig.__name__]
        
        self._patches.append(patch(full_name, replacement))
        return self

    def __enter__(self):
        for patch in self._patches:
            patch.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for patch in self._patches:
            patch.stop()
        if exc_type is not None:
            # can't really get more specific than that, we can't replace the filename of the function
            # or retrieve a previous traceback
            raise PatcherException("One of the patches failed") from exc_value


class PatcherException(Exception):
    def __init__(self, message):
        super().__init__(message)


# https://stackoverflow.com/a/19598419
def compare_ast(node1, node2):
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ("lineno", "col_offset", "ctx", "end_lineno", "end_col_offset"):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(compare_ast(n1, n2) for n1, n2 in zip(node1, node2))
    else:
        return node1 == node2


class NodeReplacer(ast.NodeTransformer):
    def __init__(self, source, target, offset, col_offset, filename, patch_traces, replace_with):
        self.source = source
        self.target = target
        self.offset = offset
        self.col_offset = col_offset
        self.filename = filename
        self.patch_traces = patch_traces
        self.replace_with = replace_with
        super().__init__()

    def visit(self, node):
        if compare_ast(node, self.source):
            instance = copy(self.target)
            kwargs = {k: getattr(node, k) for k in ["lineno", "col_offset", "end_lineno", "end_col_offset"]}
            kwargs["lineno"] += self.offset
            kwargs["end_lineno"] += self.offset
            kwargs["col_offset"] += self.col_offset
            kwargs["end_col_offset"] += self.col_offset
            self.patch_traces[(self.filename, kwargs["lineno"])] = self.replace_with
            for n in ast.walk(instance):
                for k, v in kwargs.items():
                    if hasattr(n, k):
                        setattr(n, k, v)  # d
            ast.fix_missing_locations(instance)
            return instance
        return super().visit(node)
