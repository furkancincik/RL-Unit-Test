from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from analyzer.primitive_parameter_inference import (
    infer_primitive_parameter_types,
    primitive_annotation_type,
)


_PRIMITIVE_TYPES = frozenset({"int", "float", "str", "bool"})
_CONSTRUCTOR_PREFIX = "__constructor_"
_STATE_PREFIX = "__self_"


@dataclass(frozen=True, slots=True)
class SimpleParameter:
    name: str
    type_name: str

    @property
    def analysis_name(self) -> str:
        return f"{_CONSTRUCTOR_PREFIX}{self.name}"


@dataclass(frozen=True, slots=True)
class SimpleInstanceMethodSpec:
    class_name: str
    method_name: str
    constructor_parameters: tuple[SimpleParameter, ...]
    method_parameters: tuple[SimpleParameter, ...]
    state_initializers: tuple[tuple[str, ast.expr], ...]
    method_node: ast.FunctionDef

    @property
    def qualified_name(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    @property
    def analysis_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            parameter.analysis_name
            for parameter in self.constructor_parameters
        ) + tuple(parameter.name for parameter in self.method_parameters)

    @property
    def analysis_parameter_types(self) -> dict[str, str]:
        return {
            **{
                parameter.analysis_name: parameter.type_name
                for parameter in self.constructor_parameters
            },
            **{
                parameter.name: parameter.type_name
                for parameter in self.method_parameters
            },
        }

    def split_arguments(
        self,
        values: tuple[tuple[str, object], ...],
    ) -> tuple[tuple[tuple[str, object], ...], tuple[tuple[str, object], ...]]:
        constructor: list[tuple[str, object]] = []
        method: list[tuple[str, object]] = []
        for name, value in values:
            if name.startswith(_CONSTRUCTOR_PREFIX):
                constructor.append((name[len(_CONSTRUCTOR_PREFIX) :], value))
            else:
                method.append((name, value))
        return tuple(constructor), tuple(method)


class _SelfStateTransformer(ast.NodeTransformer):
    def __init__(self, *, empty_dict_attributes: frozenset[str]) -> None:
        self._empty_dict_attributes = empty_dict_attributes

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Attribute)
            and _is_self_attribute(node.func.value)
            and node.func.value.attr in self._empty_dict_attributes
            and node.func.attr in {"items", "keys", "values"}
            and not node.args
            and not node.keywords
        ):
            replacement = ast.Tuple(elts=[], ctx=ast.Load())
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            replacement = ast.Name(
                id=f"{_STATE_PREFIX}{node.attr}",
                ctx=copy.deepcopy(node.ctx),
            )
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)


def analyze_simple_instance_method(
    class_node: ast.ClassDef,
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[SimpleInstanceMethodSpec | None, str | None]:
    if class_node.bases or class_node.keywords:
        return None, "Class inheritance and metaclass targets are unsupported."
    if class_node.decorator_list:
        return None, "Decorated classes are unsupported."
    if isinstance(method_node, ast.AsyncFunctionDef):
        return None, "Async methods are unsupported."
    if method_node.name == "__init__":
        return None, "Constructor methods are not standalone analysis targets."
    if method_node.name.startswith("__") and method_node.name.endswith("__"):
        return None, "Dunder methods are excluded from automatic production runs."
    if method_node.decorator_list:
        return None, "Decorated instance methods are unsupported."

    method_parameters, reason = _parameters(
        method_node,
        require_self=True,
        allow_local_object_annotation=True,
    )
    if reason is not None:
        return None, reason

    constructor = next(
        (
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
        ),
        None,
    )
    constructor_parameters: tuple[SimpleParameter, ...] = ()
    state_initializers: tuple[tuple[str, ast.expr], ...] = ()
    if constructor is not None:
        if isinstance(constructor, ast.AsyncFunctionDef) or constructor.decorator_list:
            return None, "The constructor shape is unsupported."
        constructor_parameters, reason = _parameters(
            constructor,
            require_self=True,
        )
        if reason is not None:
            return None, f"Unsupported constructor: {reason}"
        state_initializers, reason = _constructor_state(
            constructor,
            {parameter.name for parameter in constructor_parameters},
        )
        if reason is not None:
            return None, reason

    reason = _validate_method_state(
        method_node,
        initialized_state={
            name: _empty_collection_kind(initializer)
            for name, initializer in state_initializers
        },
        parameter_names={parameter.name for parameter in method_parameters},
    )
    if reason is not None:
        return None, reason

    return (
        SimpleInstanceMethodSpec(
            class_name=class_node.name,
            method_name=method_node.name,
            constructor_parameters=constructor_parameters,
            method_parameters=method_parameters,
            state_initializers=state_initializers,
            method_node=method_node,
        ),
        None,
    )


def normalized_method_node(spec: SimpleInstanceMethodSpec) -> ast.FunctionDef:
    method = copy.deepcopy(spec.method_node)
    transformer = _SelfStateTransformer(
        empty_dict_attributes=frozenset(
            name
            for name, initializer in spec.state_initializers
            if _empty_collection_kind(initializer) == "dict"
        )
    )
    body = [transformer.visit(statement) for statement in method.body]
    constructor_names = {
        parameter.name: parameter.analysis_name
        for parameter in spec.constructor_parameters
    }
    prelude: list[ast.stmt] = []
    for attribute_name, initializer in spec.state_initializers:
        value = copy.deepcopy(initializer)
        if isinstance(value, ast.Name) and value.id in constructor_names:
            value.id = constructor_names[value.id]
        assignment = ast.Assign(
            targets=[
                ast.Name(
                    id=f"{_STATE_PREFIX}{attribute_name}",
                    ctx=ast.Store(),
                )
            ],
            value=value,
            lineno=method.lineno,
            col_offset=method.col_offset,
            end_lineno=method.lineno,
            end_col_offset=method.col_offset,
        )
        prelude.append(assignment)

    constructor_arguments = [
        ast.arg(
            arg=parameter.analysis_name,
            annotation=ast.Name(id=parameter.type_name, ctx=ast.Load()),
        )
        for parameter in spec.constructor_parameters
    ]
    method_arguments = copy.deepcopy(method.args.args[1:])
    method.args = ast.arguments(
        posonlyargs=[],
        args=[*constructor_arguments, *method_arguments],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    method.body = [*prelude, *body]
    method.decorator_list = []
    method.name = spec.qualified_name
    return ast.fix_missing_locations(method)


def find_analysis_target(
    tree: ast.Module,
    target_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    parts = target_name.split(".")
    if len(parts) == 1:
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if statement.name == target_name:
                    if isinstance(statement, ast.AsyncFunctionDef):
                        return statement
                    from analyzer.safe_custom_object import (
                        analyze_safe_custom_object_target,
                        normalized_custom_object_target,
                    )

                    custom_spec, reason = analyze_safe_custom_object_target(
                        tree,
                        target_name,
                    )
                    if reason is not None:
                        raise ValueError(reason)
                    return (
                        normalized_custom_object_target(custom_spec)
                        if custom_spec is not None
                        else statement
                    )
        raise ValueError(f"Fonksiyon bulunamadı: {target_name}")
    if len(parts) != 2 or any(not part.isidentifier() for part in parts):
        raise ValueError(f"Geçersiz analiz hedefi: {target_name}")
    class_name, method_name = parts
    class_node = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, ast.ClassDef) and statement.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise ValueError(f"Sınıf bulunamadı: {class_name}")
    method_node = next(
        (
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == method_name
        ),
        None,
    )
    if method_node is None:
        raise ValueError(f"Method bulunamadı: {target_name}")
    spec, reason = analyze_simple_instance_method(class_node, method_node)
    if spec is None:
        raise ValueError(reason or "Instance method desteklenmiyor.")
    normalized = normalized_method_node(spec)
    from analyzer.safe_custom_object import (
        analyze_safe_custom_object_target,
        normalized_custom_object_target,
    )

    custom_spec, custom_reason = analyze_safe_custom_object_target(
        tree,
        target_name,
    )
    if custom_reason is not None:
        raise ValueError(custom_reason)
    return (
        normalized_custom_object_target(custom_spec, normalized)
        if custom_spec is not None
        else normalized
    )


def method_spec_for_target(
    tree: ast.Module,
    target_name: str,
) -> SimpleInstanceMethodSpec | None:
    parts = target_name.split(".")
    if len(parts) == 1:
        return None
    if len(parts) != 2:
        raise ValueError(f"Geçersiz analiz hedefi: {target_name}")
    class_node = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, ast.ClassDef) and statement.name == parts[0]
        ),
        None,
    )
    if class_node is None:
        raise ValueError(f"Sınıf bulunamadı: {parts[0]}")
    method_node = next(
        (
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == parts[1]
        ),
        None,
    )
    if method_node is None:
        raise ValueError(f"Method bulunamadı: {target_name}")
    spec, reason = analyze_simple_instance_method(class_node, method_node)
    if spec is None:
        raise ValueError(reason or "Instance method desteklenmiyor.")
    return spec


def _parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    require_self: bool,
    allow_local_object_annotation: bool = False,
) -> tuple[tuple[SimpleParameter, ...], str | None]:
    if node.args.posonlyargs or node.args.kwonlyargs:
        return (), "Positional-only and keyword-only parameters are unsupported."
    if node.args.vararg is not None or node.args.kwarg is not None:
        return (), "Variadic parameters are unsupported."
    arguments = list(node.args.args)
    if require_self:
        if not arguments or arguments[0].arg != "self":
            return (), "Instance methods must declare self as the first parameter."
        arguments = arguments[1:]
    inference = infer_primitive_parameter_types(
        node,
        {argument.arg for argument in arguments if argument.annotation is None},
    )
    parameters: list[SimpleParameter] = []
    for argument in arguments:
        type_name = primitive_annotation_type(argument.annotation)
        if type_name is None:
            type_name = inference.type_for(argument.arg)
        if type_name is None and allow_local_object_annotation:
            type_name = _local_object_annotation_token(argument.annotation)
        is_local_object_annotation = (
            allow_local_object_annotation
            and type_name is not None
            and type_name == _local_object_annotation_token(argument.annotation)
        )
        if type_name not in _PRIMITIVE_TYPES and not is_local_object_annotation:
            rejection = inference.rejection_for(argument.arg)
            if rejection is not None:
                return (), f"Parameter {argument.arg!r} {rejection}"
            return (), f"Parameter {argument.arg!r} is not a supported primitive."
        parameters.append(SimpleParameter(argument.arg, type_name))
    return tuple(parameters), None


def _local_object_annotation_token(annotation: ast.expr | None) -> str | None:
    if (
        isinstance(annotation, ast.Name)
        and annotation.id not in {"object", "Any", "None"}
    ):
        return annotation.id
    if (
        isinstance(annotation, ast.Constant)
        and isinstance(annotation.value, str)
        and annotation.value.isidentifier()
        and annotation.value not in {"object", "Any", "None"}
    ):
        return annotation.value
    return None


def _constructor_state(
    constructor: ast.FunctionDef,
    parameter_names: set[str],
) -> tuple[tuple[tuple[str, ast.expr], ...], str | None]:
    initializers: list[tuple[str, ast.expr]] = []
    initialized_attributes: set[str] = set()
    for statement in constructor.body:
        if _is_docstring(statement) or isinstance(statement, ast.Pass):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return (), "Unsupported constructor state statement."
        target = statement.targets[0]
        if not _is_self_attribute(target):
            return (), "Constructor may only assign direct self attributes."
        if target.attr in initialized_attributes:
            return (), "Constructor self attributes may only be initialized once."
        if not (
            _is_primitive_literal(statement.value)
            or _empty_collection_kind(statement.value) is not None
            or isinstance(statement.value, ast.Name)
            and statement.value.id in parameter_names
        ):
            return (
                (),
                "Constructor collection state must be a direct empty list or dict; "
                "other state must use a primitive literal or parameter.",
            )
        initializers.append((target.attr, copy.deepcopy(statement.value)))
        initialized_attributes.add(target.attr)
    return tuple(initializers), None


def _validate_method_state(
    method: ast.FunctionDef,
    *,
    initialized_state: dict[str, str | None],
    parameter_names: set[str],
) -> str | None:
    parents = {
        child: parent
        for parent in ast.walk(method)
        for child in ast.iter_child_nodes(parent)
    }
    assigned = set(initialized_state)
    collection_state = {
        name: kind for name, kind in initialized_state.items() if kind is not None
    }
    for node in ast.walk(method):
        if node is not method and isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            return "Nested callable or class definitions in methods are unsupported."
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return "Generator methods are unsupported."
        if isinstance(node, ast.Name) and node.id == "self":
            parent = parents.get(node)
            if not isinstance(parent, ast.Attribute) or parent.value is not node:
                return "Direct self object use is unsupported."
        if not _is_self_attribute(node):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            if _is_safe_empty_dict_view(node, parent, parents, collection_state):
                continue
            if node.attr in collection_state:
                return "Empty collection mutation or arbitrary method calls are unsupported."
            return "Nested or dynamic self attributes are unsupported."
        if isinstance(parent, ast.Call) and parent.func is node:
            return "Arbitrary instance method calls are unsupported."
        if node.attr in collection_state:
            if isinstance(node.ctx, ast.Store):
                return "Empty collection state mutation is unsupported."
            if isinstance(parent, ast.Subscript) and parent.value is node:
                return "Empty collection subscript access or mutation is unsupported."
            if _is_collection_alias(node, parent):
                return "Empty collection state aliasing is unsupported."
            if not _is_safe_empty_collection_read(node, parent, parents):
                return "Unsupported empty collection state read."
        if isinstance(node.ctx, ast.Store):
            assigned.add(node.attr)

    for node in ast.walk(method):
        if _is_self_attribute(node) and isinstance(node.ctx, ast.Load):
            if node.attr not in assigned:
                return f"Self attribute {node.attr!r} is not safely initialized."
        if isinstance(node, ast.Assign) and any(
            _is_self_attribute(target) for target in node.targets
        ):
            if len(node.targets) != 1 or not _safe_mutation_value(
                node.value, parameter_names
            ):
                return "Self state assignment must use a literal or method parameter."
        if isinstance(node, ast.AugAssign) and _is_self_attribute(node.target):
            if not isinstance(node.op, (ast.Add, ast.Sub)):
                return "Only += and -= self state mutations are supported."
            if not _safe_mutation_value(node.value, parameter_names):
                return "Self state mutation must use a literal or method parameter."
    return None


def _is_primitive_literal(value: ast.expr) -> bool:
    if isinstance(value, ast.Constant):
        return type(value.value) in (int, float, str, bool)
    return (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, (ast.UAdd, ast.USub))
        and isinstance(value.operand, ast.Constant)
        and type(value.operand.value) in (int, float)
    )


def _empty_collection_kind(value: ast.expr) -> str | None:
    if isinstance(value, ast.List) and not value.elts:
        return "list"
    if isinstance(value, ast.Dict) and not value.keys and not value.values:
        return "dict"
    return None


def _is_safe_empty_dict_view(
    node: ast.Attribute,
    parent: ast.Attribute,
    parents: dict[ast.AST, ast.AST],
    collection_state: dict[str, str],
) -> bool:
    if collection_state.get(node.attr) != "dict":
        return False
    if parent.attr not in {"items", "keys", "values"}:
        return False
    call = parents.get(parent)
    if not isinstance(call, ast.Call) or call.func is not parent:
        return False
    if call.args or call.keywords:
        return False
    loop = parents.get(call)
    return isinstance(loop, ast.For) and loop.iter is call


def _is_collection_alias(node: ast.Attribute, parent: ast.AST | None) -> bool:
    return (
        isinstance(parent, ast.Assign) and parent.value is node
        or isinstance(parent, ast.AnnAssign) and parent.value is node
        or isinstance(parent, ast.NamedExpr) and parent.value is node
    )


def _is_safe_empty_collection_read(
    node: ast.Attribute,
    parent: ast.AST | None,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    if isinstance(parent, (ast.If, ast.While, ast.IfExp)) and parent.test is node:
        return True
    if isinstance(parent, ast.BoolOp):
        return True
    if isinstance(parent, ast.Compare):
        return (
            len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and len(parent.comparators) == 1
            and parent.comparators[0] is node
        )
    if isinstance(parent, ast.For) and parent.iter is node:
        return True
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id == "bool"
        and parent.args == [node]
        and not parent.keywords
    ):
        return True
    if isinstance(parent, ast.Attribute) and parent.value is node:
        call = parents.get(parent)
        loop = parents.get(call) if call is not None else None
        return isinstance(call, ast.Call) and isinstance(loop, ast.For)
    return False


def _safe_mutation_value(value: ast.expr, parameter_names: set[str]) -> bool:
    return _is_primitive_literal(value) or (
        isinstance(value, ast.Name) and value.id in parameter_names
    )


def _is_self_attribute(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )
