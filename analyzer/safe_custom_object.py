from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

from analyzer.primitive_parameter_inference import (
    infer_primitive_parameter_types,
    primitive_annotation_type,
    primitive_literal_type,
)


MAX_SAFE_OBJECTS_PER_SCENARIO = 4
_PRIMITIVE_TYPES = frozenset({"int", "float", "str", "bool"})
_KNOWN_NON_OBJECT_ANNOTATIONS = frozenset(
    {
        *_PRIMITIVE_TYPES,
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "object",
        "Any",
        "None",
    }
)
_ANALYSIS_PREFIX = "__custom_object_"
_EMPTY_COLLECTION_KINDS = frozenset(
    {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE"}
)
_CONSTRUCTOR_EMPTY_PROOF = "CONSTRUCTOR_UNCONDITIONAL_ASSIGNMENT"
UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER = (
    "__rlut_unsupported_custom_object_method_call_7f4d3a"
)


@dataclass(frozen=True, slots=True)
class SafeConstructorParameter:
    """Doğrulanmış primitive constructor parametresidir."""

    name: str
    type_name: str
    analysis_name: str


@dataclass(frozen=True, slots=True)
class SafeStateInitializer:
    """Bir direct instance attribute'un güvenli başlangıç provenance'ıdır."""

    attribute_name: str
    constructor_parameter_name: str | None = None
    literal_value: int | float | str | bool | None = field(
        default=None,
        repr=False,
    )
    uses_literal: bool = False
    empty_collection_kind: str | None = None
    proof_origin: str | None = field(default=None, repr=False)
    proof_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.attribute_name.isidentifier():
            raise ValueError("State attribute adı geçerli identifier olmalıdır.")
        if self.empty_collection_kind is None:
            if self.proof_origin is not None:
                raise ValueError("Primitive state empty collection proof taşıyamaz.")
            object.__setattr__(self, "proof_fingerprint", "")
            return
        if self.empty_collection_kind not in _EMPTY_COLLECTION_KINDS:
            raise ValueError("Empty collection state türü desteklenmiyor.")
        if self.proof_origin != _CONSTRUCTOR_EMPTY_PROOF:
            raise ValueError("Empty collection state proof origin geçersizdir.")
        if (
            self.constructor_parameter_name is not None
            or self.uses_literal
            or self.literal_value is not None
        ):
            raise ValueError("Empty collection proof runtime state değeri taşıyamaz.")
        canonical = json.dumps(
            {
                "attribute": self.attribute_name,
                "kind": self.empty_collection_kind,
                "origin": self.proof_origin,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(
            self,
            "proof_fingerprint",
            hashlib.sha256(canonical).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class SafeObjectParameterSpec:
    """Tek bir target parametresinin local object construction şemasıdır."""

    parameter_name: str
    class_name: str
    constructor_parameters: tuple[SafeConstructorParameter, ...]
    state_initializers: tuple[SafeStateInitializer, ...]
    class_fingerprint: str
    resolution_kind: str = field(default="ANNOTATED", repr=False)
    depth: int = 1

    @property
    def state_by_name(self) -> dict[str, SafeStateInitializer]:
        return {
            initializer.attribute_name: initializer
            for initializer in self.state_initializers
        }


@dataclass(frozen=True, slots=True)
class SafeObjectConstructionBlueprint:
    """Public modellere açılmayan immutable object construction planıdır."""

    module_identity: str = field(repr=False)
    class_name: str
    constructor_arguments: tuple[tuple[str, int | float | str | bool], ...] = field(
        default=(),
        repr=False,
    )
    depth: int = 1
    class_fingerprint: str = field(default="", repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_module_identity(self.module_identity)
        if not self.class_name.isidentifier():
            raise ValueError("Blueprint class_name geçerli identifier olmalıdır.")
        if self.depth != 1:
            raise ValueError("Custom object construction depth yalnız 1 olabilir.")
        if not isinstance(self.constructor_arguments, tuple):
            raise TypeError("constructor_arguments tuple olmalıdır.")
        names: list[str] = []
        canonical_arguments: list[dict[str, Any]] = []
        for name, value in self.constructor_arguments:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Constructor parameter adı geçersizdir.")
            if type(value) not in (int, float, str, bool):
                raise TypeError("Blueprint yalnız primitive constructor değeri taşır.")
            names.append(name)
            canonical_arguments.append(
                {
                    "name": name,
                    "type": type(value).__name__,
                    "value": value,
                }
            )
        if len(set(names)) != len(names):
            raise ValueError("Constructor parameter adları tekrar edemez.")
        payload = {
            "module_identity": self.module_identity,
            "class_name": self.class_name,
            "class_fingerprint": self.class_fingerprint,
            "depth": self.depth,
            "arguments": canonical_arguments,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(
            self,
            "fingerprint",
            hashlib.sha256(canonical).hexdigest(),
        )

    def __repr__(self) -> str:
        return (
            "SafeObjectConstructionBlueprint("
            f"class_name={self.class_name!r}, depth={self.depth}, "
            f"fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True, slots=True)
class SafeCustomObjectTargetSpec:
    """Bir callable target'ın güvenli local-object parametre modelidir."""

    target_name: str
    original_parameter_names: tuple[str, ...]
    object_parameters: tuple[SafeObjectParameterSpec, ...]
    target_node: ast.FunctionDef = field(repr=False, compare=False)

    @property
    def analysis_parameter_names(self) -> tuple[str, ...]:
        names, _ = self.analysis_metadata(
            self.original_parameter_names,
            {},
        )
        return names

    def analysis_metadata(
        self,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        bindings = {
            parameter.parameter_name: parameter
            for parameter in self.object_parameters
        }
        names: list[str] = []
        types: dict[str, str] = {}
        for name in parameter_names:
            binding = bindings.get(name)
            if binding is None:
                names.append(name)
                if name in parameter_types:
                    types[name] = parameter_types[name]
                continue
            for constructor_parameter in binding.constructor_parameters:
                names.append(constructor_parameter.analysis_name)
                types[constructor_parameter.analysis_name] = (
                    constructor_parameter.type_name
                )
        return tuple(names), types

    def bind_scenario(
        self,
        scenario: Any,
        *,
        module_identity: str,
    ) -> Any:
        """Synthetic primitive girdileri immutable object blueprint'e bağlar."""
        from generator.scenario_generator import Scenario

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario Scenario olmalıdır.")
        _validate_module_identity(module_identity)
        values = dict(scenario.keyword_arguments)
        bindings = {
            parameter.parameter_name: parameter
            for parameter in self.object_parameters
        }
        bound: list[tuple[str, Any]] = []
        for parameter_name in self.original_parameter_names:
            binding = bindings.get(parameter_name)
            if binding is None:
                if parameter_name not in values:
                    raise ValueError(
                        f"Scenario parameter değeri bulunamadı: {parameter_name}"
                    )
                bound.append((parameter_name, values[parameter_name]))
                continue
            arguments: list[tuple[str, int | float | str | bool]] = []
            for constructor_parameter in binding.constructor_parameters:
                analysis_name = constructor_parameter.analysis_name
                if analysis_name not in values:
                    raise ValueError(
                        "Custom object constructor input değeri bulunamadı."
                    )
                value = values[analysis_name]
                if type(value) not in (int, float, str, bool):
                    raise TypeError(
                        "Custom object constructor input primitive olmalıdır."
                    )
                arguments.append((constructor_parameter.name, value))
            bound.append(
                (
                    parameter_name,
                    SafeObjectConstructionBlueprint(
                        module_identity=module_identity,
                        class_name=binding.class_name,
                        constructor_arguments=tuple(arguments),
                        depth=binding.depth,
                        class_fingerprint=binding.class_fingerprint,
                    ),
                )
            )
        return replace(scenario, keyword_arguments=tuple(bound))


@dataclass(frozen=True, slots=True)
class _StructuralMethodCall:
    name: str
    positional_argument_count: int
    keyword_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class _StructuralObjectInterface:
    attribute_names: frozenset[str]
    method_calls: frozenset[_StructuralMethodCall]


class _DirectInterfaceMasker(ast.NodeTransformer):
    def __init__(self, parameter_names: frozenset[str]) -> None:
        self._parameter_names = parameter_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        rewritten = self.generic_visit(node)
        assert isinstance(rewritten, ast.Attribute)
        if (
            isinstance(rewritten.value, ast.Name)
            and rewritten.value.id in self._parameter_names
        ):
            rewritten.value = ast.copy_location(
                ast.Name(id="__structural_receiver", ctx=ast.Load()),
                rewritten.value,
            )
        return rewritten


def _validate_module_identity(module_identity: str) -> None:
    if not isinstance(module_identity, str):
        raise TypeError("module_identity string olmalıdır.")
    if not module_identity or module_identity != module_identity.strip():
        raise ValueError("module_identity canonical dotted ad olmalıdır.")
    if any(
        not part or not part.isidentifier()
        for part in module_identity.split(".")
    ):
        raise ValueError("module_identity canonical dotted ad olmalıdır.")


class _ObjectStateTransformer(ast.NodeTransformer):
    def __init__(self, spec: SafeCustomObjectTargetSpec) -> None:
        self._bindings = {
            parameter.parameter_name: parameter
            for parameter in spec.object_parameters
        }

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self._bindings
        ):
            replacement = ast.Call(
                func=ast.Name(
                    id=UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER,
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self._bindings:
            binding = self._bindings[node.value.id]
            initializer = binding.state_by_name.get(node.attr)
            if initializer is None:
                return self.generic_visit(node)
            if initializer.constructor_parameter_name is not None:
                constructor_parameter = next(
                    parameter
                    for parameter in binding.constructor_parameters
                    if parameter.name == initializer.constructor_parameter_name
                )
                replacement: ast.expr = ast.Name(
                    id=constructor_parameter.analysis_name,
                    ctx=copy.deepcopy(node.ctx),
                )
            elif initializer.empty_collection_kind is not None:
                replacement = _empty_collection_expression(
                    initializer.empty_collection_kind,
                )
                if isinstance(replacement, (ast.List, ast.Tuple)):
                    replacement.ctx = copy.deepcopy(node.ctx)
            else:
                replacement = ast.Constant(value=initializer.literal_value)
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)


def analyze_safe_custom_object_target(
    tree: ast.Module,
    target_name: str,
) -> tuple[SafeCustomObjectTargetSpec | None, str | None]:
    """Local annotated object parametrelerini yalnız AST üzerinden doğrular."""
    target, is_method = _raw_target(tree, target_name)
    arguments = list(target.args.args)
    if is_method and arguments and arguments[0].arg == "self":
        arguments = arguments[1:]
    local_classes = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, ast.ClassDef)
    }
    untyped_parameter_names = frozenset(
        argument.arg for argument in arguments if argument.annotation is None
    )
    primitive_target = _DirectInterfaceMasker(
        untyped_parameter_names,
    ).visit(copy.deepcopy(target))
    assert isinstance(primitive_target, ast.FunctionDef)
    primitive_inference = infer_primitive_parameter_types(
        primitive_target,
        set(untyped_parameter_names),
    )
    object_arguments: list[tuple[ast.arg, ast.ClassDef, str]] = []
    for argument in arguments:
        resolved = _resolved_local_class(argument.annotation, local_classes)
        if resolved is not None:
            object_arguments.append((argument, resolved, "ANNOTATED"))
            continue
        annotation_reason = _unsupported_annotation_reason(
            argument.annotation,
            local_classes=frozenset(local_classes),
        )
        if annotation_reason is not None:
            return None, annotation_reason
        if argument.annotation is not None:
            continue
        if (
            primitive_inference.type_for(argument.arg) is not None
            or primitive_inference.rejection_for(argument.arg)
            == "has conflicting primitive evidence."
        ):
            continue
        interface, interface_reason = _structural_object_interface(
            target,
            argument.arg,
        )
        if interface_reason is not None:
            return None, interface_reason
        if interface is None:
            continue
        matches: list[ast.ClassDef] = []
        object_index = len(object_arguments) + 1
        for class_node in local_classes.values():
            analyzed, _ = _analyze_safe_class(
                class_node,
                object_index=object_index,
            )
            if analyzed is not None and _matches_structural_interface(
                class_node,
                analyzed,
                interface,
            ):
                matches.append(class_node)
        if not matches:
            continue
        if len(matches) != 1:
            return (
                None,
                "Untyped structural object parameter has an ambiguous safe local class match.",
            )
        object_arguments.append((argument, matches[0], "STRUCTURAL_UNIQUE"))

    if not object_arguments:
        return None, None
    if len(object_arguments) > MAX_SAFE_OBJECTS_PER_SCENARIO:
        return (
            None,
            "Custom object parameter count exceeds the bounded scenario limit.",
        )
    if _returns_local_custom_object(target.returns, frozenset(local_classes)):
        return None, "Custom object return values are unsupported."

    object_parameters: list[SafeObjectParameterSpec] = []
    occupied_analysis_names = {argument.arg for argument in arguments}
    for object_index, (argument, class_node, resolution_kind) in enumerate(
        object_arguments,
        start=1,
    ):
        analyzed, reason = _analyze_safe_class(
            class_node,
            object_index=object_index,
        )
        if reason is not None:
            return None, reason
        assert analyzed is not None
        constructor_parameters: list[SafeConstructorParameter] = []
        for parameter_index, constructor_parameter in enumerate(
            analyzed.constructor_parameters,
            start=1,
        ):
            candidate = (
                f"{_ANALYSIS_PREFIX}{object_index:03d}_{parameter_index:03d}"
            )
            while candidate in occupied_analysis_names:
                candidate += "_"
            occupied_analysis_names.add(candidate)
            constructor_parameters.append(
                replace(
                    constructor_parameter,
                    analysis_name=candidate,
                )
            )
        parameter_spec = replace(
            analyzed,
            parameter_name=argument.arg,
            constructor_parameters=tuple(constructor_parameters),
            resolution_kind=resolution_kind,
        )
        usage_reason = _validate_target_usage(
            target,
            parameter_spec,
        )
        if usage_reason is not None:
            return None, usage_reason
        object_parameters.append(parameter_spec)

    return (
        SafeCustomObjectTargetSpec(
            target_name=target_name,
            original_parameter_names=tuple(argument.arg for argument in arguments),
            object_parameters=tuple(object_parameters),
            target_node=target,
        ),
        None,
    )


def normalized_custom_object_target(
    spec: SafeCustomObjectTargetSpec,
    base_node: ast.FunctionDef | None = None,
) -> ast.FunctionDef:
    """Object state erişimlerini primitive synthetic analysis input'una indirger."""
    if not isinstance(spec, SafeCustomObjectTargetSpec):
        raise TypeError("spec SafeCustomObjectTargetSpec olmalıdır.")
    node = copy.deepcopy(base_node if base_node is not None else spec.target_node)
    if not isinstance(node, ast.FunctionDef):
        raise TypeError("Custom object target normal FunctionDef olmalıdır.")
    transformer = _ObjectStateTransformer(spec)
    node.body = [transformer.visit(statement) for statement in node.body]
    bindings = {
        binding.parameter_name: binding for binding in spec.object_parameters
    }
    rewritten_arguments: list[ast.arg] = []
    for argument in node.args.args:
        binding = bindings.get(argument.arg)
        if binding is None:
            rewritten_arguments.append(argument)
            continue
        rewritten_arguments.extend(
            ast.arg(
                arg=parameter.analysis_name,
                annotation=ast.Name(id=parameter.type_name, ctx=ast.Load()),
            )
            for parameter in binding.constructor_parameters
        )
    node.args = ast.arguments(
        posonlyargs=[],
        args=rewritten_arguments,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    return ast.fix_missing_locations(node)


def materialize_safe_blueprint(
    blueprint: SafeObjectConstructionBlueprint,
    module: ModuleType,
) -> object:
    """Trusted-dynamic aşamada doğrulanmış local class instance'ı üretir."""
    if not isinstance(blueprint, SafeObjectConstructionBlueprint):
        raise TypeError("blueprint SafeObjectConstructionBlueprint olmalıdır.")
    if not isinstance(module, ModuleType):
        raise TypeError("module ModuleType olmalıdır.")
    runtime_class = getattr(module, blueprint.class_name, None)
    if runtime_class is None:
        raise RuntimeError("Blueprint local class target modülde bulunamadı.")
    if not isinstance(runtime_class, type):
        raise TypeError("Blueprint runtime target bir class olmalıdır.")
    return runtime_class(**copy.deepcopy(dict(blueprint.constructor_arguments)))


def _raw_target(
    tree: ast.Module,
    target_name: str,
) -> tuple[ast.FunctionDef, bool]:
    parts = target_name.split(".")
    if len(parts) == 1:
        target = next(
            (
                statement
                for statement in tree.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == target_name
            ),
            None,
        )
        if target is None:
            raise ValueError(f"Fonksiyon bulunamadı: {target_name}")
        return target, False
    if len(parts) != 2 or any(not part.isidentifier() for part in parts):
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
    target = next(
        (
            statement
            for statement in class_node.body
            if isinstance(statement, ast.FunctionDef) and statement.name == parts[1]
        ),
        None,
    )
    if target is None:
        raise ValueError(f"Method bulunamadı: {target_name}")
    return target, True


def _resolved_local_class(
    annotation: ast.expr | None,
    local_classes: dict[str, ast.ClassDef],
) -> ast.ClassDef | None:
    if isinstance(annotation, ast.Name):
        return local_classes.get(annotation.id)
    if (
        isinstance(annotation, ast.Constant)
        and isinstance(annotation.value, str)
        and annotation.value.isidentifier()
    ):
        return local_classes.get(annotation.value)
    return None


def _unsupported_annotation_reason(
    annotation: ast.expr | None,
    *,
    local_classes: frozenset[str],
) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        if annotation.id in _KNOWN_NON_OBJECT_ANNOTATIONS:
            return None
        return "Custom object annotation must resolve to a local top-level class."
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        if annotation.value in _KNOWN_NON_OBJECT_ANNOTATIONS:
            return None
        return "Forward custom object annotation must resolve to a local top-level class."
    if any(isinstance(node, ast.Attribute) for node in ast.walk(annotation)):
        return "Imported or attribute custom object annotations are unsupported."
    if any(
        isinstance(node, ast.Name) and node.id in local_classes
        for node in ast.walk(annotation)
    ):
        return "Union and generic custom object annotations are unsupported."
    return None


def _structural_object_interface(
    target: ast.FunctionDef,
    parameter_name: str,
) -> tuple[_StructuralObjectInterface | None, str | None]:
    parents = {
        child: parent
        for parent in ast.walk(target)
        for child in ast.iter_child_nodes(parent)
    }
    attributes: set[str] = set()
    method_calls: set[_StructuralMethodCall] = set()
    has_direct_attribute_use = False
    has_non_interface_use = False
    for node in ast.walk(target):
        if not isinstance(node, ast.Name) or node.id != parameter_name:
            continue
        parent = parents.get(node)
        if not isinstance(parent, ast.Attribute) or parent.value is not node:
            has_non_interface_use = True
            continue
        has_direct_attribute_use = True
        current: ast.AST | None = parent
        while current is not None and current is not target:
            current = parents.get(current)
            if current is not None and current is not target and isinstance(
                current,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                return (
                    None,
                    "Untyped structural object use inside nested scopes is unsupported.",
                )
        attribute_parent = parents.get(parent)
        if isinstance(attribute_parent, ast.Call) and attribute_parent.func is parent:
            if any(isinstance(argument, ast.Starred) for argument in attribute_parent.args):
                return (
                    None,
                    "Untyped structural callable evidence may not use starred arguments.",
                )
            if any(keyword.arg is None for keyword in attribute_parent.keywords):
                return (
                    None,
                    "Untyped structural callable evidence may not use dynamic keywords.",
                )
            method_calls.add(
                _StructuralMethodCall(
                    name=parent.attr,
                    positional_argument_count=len(attribute_parent.args),
                    keyword_names=frozenset(
                        keyword.arg
                        for keyword in attribute_parent.keywords
                        if keyword.arg is not None
                    ),
                )
            )
        else:
            if not isinstance(parent.ctx, ast.Load):
                return (
                    None,
                    "Untyped structural attribute write or delete is unsupported.",
                )
            attributes.add(parent.attr)
    if not has_direct_attribute_use:
        return None, None
    if has_non_interface_use:
        return (
            None,
            "Untyped structural object parameter may only use a direct instance interface.",
        )
    return (
        _StructuralObjectInterface(
            attribute_names=frozenset(attributes),
            method_calls=frozenset(method_calls),
        ),
        None,
    )


def _matches_structural_interface(
    class_node: ast.ClassDef,
    analyzed: SafeObjectParameterSpec,
    interface: _StructuralObjectInterface,
) -> bool:
    state_names = frozenset(analyzed.state_by_name)
    methods = {
        statement.name: statement
        for statement in class_node.body
        if isinstance(statement, ast.FunctionDef)
        and statement.args.args
        and statement.args.args[0].arg == "self"
        and not statement.decorator_list
    }
    return (
        interface.attribute_names.issubset(state_names)
        and all(
            (method := methods.get(call.name)) is not None
            and _method_signature_accepts(method, call)
            for call in interface.method_calls
        )
    )


def _method_signature_accepts(
    method: ast.FunctionDef,
    call: _StructuralMethodCall,
) -> bool:
    arguments = method.args
    if (
        arguments.posonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
    ):
        return False
    positional = list(arguments.args[1:])
    if call.positional_argument_count > len(positional):
        return False
    bound_names = {
        argument.arg
        for argument in positional[: call.positional_argument_count]
    }
    positional_by_name = {argument.arg for argument in positional}
    keyword_only_by_name = {
        argument.arg for argument in arguments.kwonlyargs
    }
    for keyword_name in call.keyword_names:
        if keyword_name in bound_names:
            return False
        if (
            keyword_name not in positional_by_name
            and keyword_name not in keyword_only_by_name
        ):
            return False
        bound_names.add(keyword_name)
    required_positional_count = len(positional) - len(arguments.defaults)
    if any(
        argument.arg not in bound_names
        for argument in positional[:required_positional_count]
    ):
        return False
    return all(
        default is not None or argument.arg in bound_names
        for argument, default in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
            strict=True,
        )
    )


def _analyze_safe_class(
    class_node: ast.ClassDef,
    *,
    object_index: int,
) -> tuple[SafeObjectParameterSpec | None, str | None]:
    if class_node.bases or class_node.keywords:
        return None, "Custom object class inheritance and metaclass are unsupported."
    if class_node.decorator_list:
        return None, "Decorated custom object classes are unsupported."
    for statement in class_node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name == "__new__":
            return None, "Custom object __new__ construction is unsupported."
        if any(
            _decorator_leaf_name(decorator) == "abstractmethod"
            for decorator in statement.decorator_list
        ):
            return None, "Abstract custom object classes are unsupported."
        if any(
            _decorator_leaf_name(decorator)
            in {"property", "classmethod", "staticmethod"}
            for decorator in statement.decorator_list
        ):
            return None, "Descriptor and alternate-bound custom object methods are unsupported."
    constructor = next(
        (
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        fingerprint = _class_fingerprint(class_node.name, (), ())
        return (
            SafeObjectParameterSpec(
                parameter_name="",
                class_name=class_node.name,
                constructor_parameters=(),
                state_initializers=(),
                class_fingerprint=fingerprint,
            ),
            None,
        )
    if isinstance(constructor, ast.AsyncFunctionDef) or constructor.decorator_list:
        return None, "Custom object constructor shape is unsupported."
    if (
        constructor.args.posonlyargs
        or constructor.args.kwonlyargs
        or constructor.args.vararg is not None
        or constructor.args.kwarg is not None
    ):
        return None, "Custom object constructor signature is unsupported."
    arguments = list(constructor.args.args)
    if not arguments or arguments[0].arg != "self":
        return None, "Custom object constructor must declare self first."
    arguments = arguments[1:]
    defaults = _defaults_by_name(constructor)
    for default in defaults.values():
        if primitive_literal_type(default) is None:
            return None, "Custom object constructor defaults must be primitive literals."
    inference = infer_primitive_parameter_types(
        constructor,
        {argument.arg for argument in arguments if argument.annotation is None},
    )
    parameters: list[SafeConstructorParameter] = []
    for parameter_index, argument in enumerate(arguments, start=1):
        type_name = primitive_annotation_type(argument.annotation)
        if type_name is None:
            type_name = inference.type_for(argument.arg)
        if type_name not in _PRIMITIVE_TYPES:
            return (
                None,
                "Custom object constructor parameters must have synthesizable "
                "primitive evidence.",
            )
        parameters.append(
            SafeConstructorParameter(
                name=argument.arg,
                type_name=type_name,
                analysis_name=(
                    f"{_ANALYSIS_PREFIX}{object_index:03d}_{parameter_index:03d}"
                ),
            )
        )
    parameter_names = {parameter.name for parameter in parameters}
    initializers: list[SafeStateInitializer] = []
    initialized: set[str] = set()
    for statement in constructor.body:
        if _is_docstring(statement) or isinstance(statement, ast.Pass):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None, "Custom object constructor contains an unsafe statement."
        target = statement.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and isinstance(target.ctx, ast.Store)
        ):
            return None, "Custom object constructor may only assign direct self state."
        if target.attr in initialized:
            return None, "Custom object constructor state may only be assigned once."
        if isinstance(statement.value, ast.Name) and statement.value.id in parameter_names:
            initializer = SafeStateInitializer(
                attribute_name=target.attr,
                constructor_parameter_name=statement.value.id,
            )
        else:
            empty_collection_kind = _empty_collection_kind(statement.value)
            if empty_collection_kind is not None:
                initializer = SafeStateInitializer(
                    attribute_name=target.attr,
                    empty_collection_kind=empty_collection_kind,
                    proof_origin=_CONSTRUCTOR_EMPTY_PROOF,
                )
            else:
                literal_type = primitive_literal_type(statement.value)
                if literal_type is None:
                    return None, "Custom object constructor state must use primitive provenance."
                initializer = SafeStateInitializer(
                    attribute_name=target.attr,
                    literal_value=_literal_value(statement.value),
                    uses_literal=True,
                )
        initializers.append(initializer)
        initialized.add(target.attr)
    fingerprint = _class_fingerprint(
        class_node.name,
        tuple(parameters),
        tuple(initializers),
    )
    return (
        SafeObjectParameterSpec(
            parameter_name="",
            class_name=class_node.name,
            constructor_parameters=tuple(parameters),
            state_initializers=tuple(initializers),
            class_fingerprint=fingerprint,
        ),
        None,
    )


def _validate_target_usage(
    target: ast.FunctionDef,
    parameter: SafeObjectParameterSpec,
) -> str | None:
    parents = {
        child: parent
        for parent in ast.walk(target)
        for child in ast.iter_child_nodes(parent)
    }
    state_names = frozenset(parameter.state_by_name)
    for node in ast.walk(target):
        if isinstance(node, ast.Name) and node.id == parameter.parameter_name:
            parent = parents.get(node)
            is_path_scoped_method_call = (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and isinstance(parents.get(parent), ast.Call)
                and parents[parent].func is parent
            )
            if not (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr in state_names
            ) and not is_path_scoped_method_call:
                return (
                    "Custom object parameter may only access validated direct "
                    "primitive state."
                )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == parameter.parameter_name
        ):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            if node.attr not in state_names:
                return "Custom object dynamic or unknown attributes are unsupported."
            if isinstance(parent, ast.Attribute) and parent.value is node:
                return "Nested custom object attributes are unsupported."
            if isinstance(parent, ast.Subscript) and parent.value is node:
                return "Custom object state subscript access is unsupported."
            initializer = parameter.state_by_name[node.attr]
            if initializer.empty_collection_kind is not None:
                if not (
                    isinstance(node.ctx, ast.Load)
                    and isinstance(parent, ast.UnaryOp)
                    and isinstance(parent.op, ast.Not)
                    and parent.operand is node
                ):
                    return "Unsupported constructor-proven empty collection state read."
            if (
                isinstance(node.ctx, ast.Store)
                and (
                    initializer.uses_literal
                    or initializer.empty_collection_kind is not None
                )
            ):
                return "Literal-backed custom object state mutation is unsupported."
    return None


def _returns_local_custom_object(
    annotation: ast.expr | None,
    local_classes: frozenset[str],
) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in local_classes
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value in local_classes
    return False


def _defaults_by_name(constructor: ast.FunctionDef) -> dict[str, ast.expr]:
    arguments = constructor.args.args
    if not constructor.args.defaults:
        return {}
    return {
        argument.arg: default
        for argument, default in zip(
            arguments[-len(constructor.args.defaults) :],
            constructor.args.defaults,
            strict=True,
        )
    }


def _literal_value(value: ast.expr) -> int | float | str | bool:
    if isinstance(value, ast.Constant) and type(value.value) in (int, float, str, bool):
        return value.value
    if (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, (ast.UAdd, ast.USub))
        and isinstance(value.operand, ast.Constant)
        and type(value.operand.value) in (int, float)
    ):
        return +value.operand.value if isinstance(value.op, ast.UAdd) else -value.operand.value
    raise TypeError("Primitive literal bekleniyordu.")


def _empty_collection_kind(value: ast.expr) -> str | None:
    if isinstance(value, ast.List) and not value.elts:
        return "EMPTY_LIST"
    if isinstance(value, ast.Dict) and not value.keys and not value.values:
        return "EMPTY_DICT"
    if isinstance(value, ast.Tuple) and not value.elts:
        return "EMPTY_TUPLE"
    return None


def _empty_collection_expression(kind: str) -> ast.expr:
    if kind == "EMPTY_LIST":
        return ast.List(elts=[], ctx=ast.Load())
    if kind == "EMPTY_DICT":
        return ast.Dict(keys=[], values=[])
    if kind == "EMPTY_TUPLE":
        return ast.Tuple(elts=[], ctx=ast.Load())
    raise ValueError("Empty collection state türü desteklenmiyor.")


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _decorator_leaf_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _class_fingerprint(
    class_name: str,
    parameters: tuple[SafeConstructorParameter, ...],
    initializers: tuple[SafeStateInitializer, ...],
) -> str:
    payload = {
        "class_name": class_name,
        "parameters": [
            {"name": parameter.name, "type": parameter.type_name}
            for parameter in parameters
        ],
        "state": [
            {
                "attribute": initializer.attribute_name,
                "parameter": initializer.constructor_parameter_name,
                "literal_type": (
                    type(initializer.literal_value).__name__
                    if initializer.uses_literal
                    else None
                ),
                "literal": (
                    initializer.literal_value if initializer.uses_literal else None
                ),
                "empty_collection_kind": initializer.empty_collection_kind,
                "proof_origin": initializer.proof_origin,
                "proof_fingerprint": initializer.proof_fingerprint,
            }
            for initializer in initializers
        ],
        "depth": 1,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
