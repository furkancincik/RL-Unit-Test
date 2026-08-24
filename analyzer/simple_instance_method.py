from __future__ import annotations

import ast
import copy
from dataclasses import dataclass


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

    method_parameters, reason = _parameters(method_node, require_self=True)
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
        initialized_attributes={name for name, _ in state_initializers},
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
    transformer = _SelfStateTransformer()
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
                    return statement
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
    return normalized_method_node(spec)


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
    defaults_by_name: dict[str, ast.expr] = {}
    if node.args.defaults:
        for argument, default in zip(
            arguments[-len(node.args.defaults) :],
            node.args.defaults,
            strict=True,
        ):
            defaults_by_name[argument.arg] = default
    parameters: list[SimpleParameter] = []
    for argument in arguments:
        type_name = _primitive_annotation(argument.annotation)
        if type_name is None:
            type_name = _literal_type(defaults_by_name.get(argument.arg))
        if type_name not in _PRIMITIVE_TYPES:
            return (), f"Parameter {argument.arg!r} is not a supported primitive."
        parameters.append(SimpleParameter(argument.arg, type_name))
    return tuple(parameters), None


def _constructor_state(
    constructor: ast.FunctionDef,
    parameter_names: set[str],
) -> tuple[tuple[tuple[str, ast.expr], ...], str | None]:
    initializers: list[tuple[str, ast.expr]] = []
    for statement in constructor.body:
        if _is_docstring(statement) or isinstance(statement, ast.Pass):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return (), "Unsupported constructor state statement."
        target = statement.targets[0]
        if not _is_self_attribute(target):
            return (), "Constructor may only assign direct self attributes."
        if not (
            _is_primitive_literal(statement.value)
            or isinstance(statement.value, ast.Name)
            and statement.value.id in parameter_names
        ):
            return (), "Constructor state must use a literal or primitive parameter."
        initializers.append((target.attr, copy.deepcopy(statement.value)))
    return tuple(initializers), None


def _validate_method_state(
    method: ast.FunctionDef,
    *,
    initialized_attributes: set[str],
    parameter_names: set[str],
) -> str | None:
    parents = {
        child: parent
        for parent in ast.walk(method)
        for child in ast.iter_child_nodes(parent)
    }
    assigned = set(initialized_attributes)
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
            return "Nested or dynamic self attributes are unsupported."
        if isinstance(parent, ast.Call) and parent.func is node:
            return "Arbitrary instance method calls are unsupported."
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


def _primitive_annotation(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name) and annotation.id in _PRIMITIVE_TYPES:
        return annotation.id
    return None


def _literal_type(value: ast.expr | None) -> str | None:
    if isinstance(value, ast.Constant) and type(value.value) in (int, float, str, bool):
        return type(value.value).__name__
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
