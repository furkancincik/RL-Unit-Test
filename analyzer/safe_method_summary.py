from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


SUMMARY_VERSION = 1
_PRIMITIVE_ORDER = ("bool", "int", "float", "str")
_PRIMITIVE_TYPES = frozenset(_PRIMITIVE_ORDER)
_NUMERIC_TYPES = frozenset({"int", "float"})
_HASHABLE_PRIMITIVE_TYPES = frozenset(_PRIMITIVE_TYPES)


class SafeMethodCategory(str, Enum):
    MUTATOR = "MUTATOR"
    OBSERVER = "OBSERVER"
    COMPOSED = "COMPOSED"
    UNSUPPORTED = "UNSUPPORTED"


class SafeMethodRejectionCategory(str, Enum):
    DOMAIN_CONFLICT = "DOMAIN_CONFLICT"
    UNKNOWN_PRIMITIVE_DOMAIN = "UNKNOWN_PRIMITIVE_DOMAIN"
    UNSAFE_CONSTRUCTOR_STATE = "UNSAFE_CONSTRUCTOR_STATE"
    UNSAFE_CALL = "UNSAFE_CALL"
    DYNAMIC_IMPORT = "DYNAMIC_IMPORT"
    IMPORT_OR_PROCESS = "IMPORT_OR_PROCESS"
    GLOBAL_MUTATION = "GLOBAL_MUTATION"
    REFLECTION = "REFLECTION"
    UNSAFE_RECEIVER = "UNSAFE_RECEIVER"
    DELETE = "DELETE"
    EXCEPTION_DEPENDENT_EFFECT = "EXCEPTION_DEPENDENT_EFFECT"
    UNBOUNDED_CONTROL_FLOW = "UNBOUNDED_CONTROL_FLOW"
    NESTED_SCOPE_MUTATION = "NESTED_SCOPE_MUTATION"
    UNKNOWN_OBJECT_GRAPH = "UNKNOWN_OBJECT_GRAPH"
    CALLBACK = "CALLBACK"
    GENERATOR = "GENERATOR"
    CONTEXT_MANAGER = "CONTEXT_MANAGER"
    DECORATED_METHOD = "DECORATED_METHOD"
    STATIC_OR_CLASS_METHOD = "STATIC_OR_CLASS_METHOD"
    DYNAMIC_DISPATCH = "DYNAMIC_DISPATCH"
    INHERITANCE_OR_METACLASS = "INHERITANCE_OR_METACLASS"
    ASYNC_METHOD = "ASYNC_METHOD"
    CALL_CYCLE = "CALL_CYCLE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    UNPROVEN_LOOKUP = "UNPROVEN_LOOKUP"
    UNKNOWN_METHOD = "UNKNOWN_METHOD"
    UNSAFE_CONSTRUCT = "UNSAFE_CONSTRUCT"
    UNSAFE_SIGNATURE = "UNSAFE_SIGNATURE"
    UNSAFE_CLASS_NAMESPACE = "UNSAFE_CLASS_NAMESPACE"


@dataclass(frozen=True, slots=True)
class SafeMethodBudgets:
    max_objects: int = 4
    constructor_depth: int = 1
    setup_relation_depth: int = 2
    top_level_setup_calls: int = 2
    repeated_mutator_calls: int = 2
    collection_cardinality: int = 2
    local_call_depth: int = 1
    guards: int = 3
    state_effects: int = 2
    observer_calls: int = 2
    executable_statements: int = 7

    def __post_init__(self) -> None:
        for value in self.canonical_payload:
            if type(value) is not int or value < 1:
                raise ValueError("Method summary budgets pozitif tam sayı olmalıdır.")

    @property
    def canonical_payload(self) -> tuple[int, ...]:
        return (
            self.max_objects,
            self.constructor_depth,
            self.setup_relation_depth,
            self.top_level_setup_calls,
            self.repeated_mutator_calls,
            self.collection_cardinality,
            self.local_call_depth,
            self.guards,
            self.state_effects,
            self.observer_calls,
            self.executable_statements,
        )


DEFAULT_SAFE_METHOD_BUDGETS = SafeMethodBudgets()


@dataclass(frozen=True, slots=True)
class SafePrimitiveDomain:
    admissible_types: tuple[str, ...]
    evidence_kinds: tuple[str, ...] = ()
    canonical_value: int | float | str | bool | None = field(
        default=None,
        repr=False,
    )
    allows_nan: bool = False
    allows_infinity: bool = False

    def __post_init__(self) -> None:
        _require_tuple(self.admissible_types, "admissible_types")
        normalized = _ordered_types(self.admissible_types)
        if normalized != self.admissible_types or not normalized:
            raise ValueError("Primitive domain canonical ve boş olmayan olmalıdır.")
        if self.canonical_value is not None:
            value_type = type(self.canonical_value).__name__
            if value_type not in self.admissible_types:
                raise ValueError("Canonical primitive değer domain ile uyumsuzdur.")
            if isinstance(self.canonical_value, float) and not math.isfinite(
                self.canonical_value
            ):
                raise ValueError("NaN ve infinity canonical primitive olamaz.")
        if self.allows_nan or self.allows_infinity:
            raise ValueError("Safe primitive domain özel float değerleri kabul etmez.")
        if type(self.allows_nan) is not bool or type(self.allows_infinity) is not bool:
            raise TypeError("Float domain bayrakları bool olmalıdır.")
        object.__setattr__(
            self,
            "evidence_kinds",
            tuple(sorted(set(self.evidence_kinds))),
        )

    @property
    def is_exact(self) -> bool:
        return len(self.admissible_types) == 1

    @property
    def canonical_payload(self) -> tuple[Any, ...]:
        return (
            self.admissible_types,
            self.evidence_kinds,
            _canonical_primitive(self.canonical_value),
            self.allows_nan,
            self.allows_infinity,
        )


@dataclass(frozen=True, slots=True)
class SafeReceiverProof:
    module_identity: str
    class_identity: str
    method_identity: str
    class_fingerprint: str
    class_semantic_digest: str
    direct_self: bool = True
    class_namespace_safe: bool = field(default=True, repr=False)
    constructor_safe: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        _validate_module_identity(self.module_identity)
        for field_name, value in (
            ("class_identity", self.class_identity),
            ("method_identity", self.method_identity),
        ):
            if type(value) is not str or not value.isidentifier():
                raise ValueError(f"{field_name} geçerli identifier olmalıdır.")
        for field_name, value in (
            ("class_fingerprint", self.class_fingerprint),
            ("class_semantic_digest", self.class_semantic_digest),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} canonical SHA-256 olmalıdır.")
        if type(self.direct_self) is not bool:
            raise TypeError("direct_self bool olmalıdır.")
        if type(self.class_namespace_safe) is not bool:
            raise TypeError("class_namespace_safe bool olmalıdır.")
        if type(self.constructor_safe) is not bool:
            raise TypeError("constructor_safe bool olmalıdır.")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.module_identity,
            self.class_identity,
            self.method_identity,
            self.class_fingerprint,
            self.direct_self,
            self.class_namespace_safe,
            self.constructor_safe,
        )

    @property
    def canonical_semantic_payload(self) -> tuple[Any, ...]:
        return (
            self.class_semantic_digest,
            self.direct_self,
            self.class_namespace_safe,
            self.constructor_safe,
        )


@dataclass(frozen=True, slots=True)
class SafeGuardSummary:
    kind: str
    operator: str | None
    branch_polarities: tuple[bool, ...]
    narrowed_domains: tuple[tuple[str, tuple[str, ...]], ...] = ()
    literal_values: tuple[tuple[str, int | float | str | bool], ...] = field(
        default=(),
        repr=False,
    )
    expression_shape: tuple[Any, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _require_string(self.kind, "kind")
        _require_optional_string(self.operator, "operator")
        _require_deep_immutable(self.branch_polarities, "branch_polarities")
        if not self.branch_polarities or any(
            type(item) is not bool for item in self.branch_polarities
        ):
            raise TypeError("branch_polarities yalnız bool değerler taşımalıdır.")
        _require_deep_immutable(self.narrowed_domains, "narrowed_domains")
        guard_domain_order = (
            *_PRIMITIVE_ORDER,
            "EMPTY_LIST",
            "EMPTY_DICT",
            "EMPTY_TUPLE",
        )
        for item in self.narrowed_domains:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not tuple
                or not item[1]
                or item[1]
                != tuple(
                    candidate
                    for candidate in guard_domain_order
                    if candidate in item[1]
                )
            ):
                raise ValueError("narrowed_domains canonical domain taşımalıdır.")
        _require_deep_immutable(self.literal_values, "literal_values")
        for item in self.literal_values:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("literal_values canonical tuple taşımalıdır.")
            type_name, value = item
            if type(value) not in (int, float, str, bool):
                raise TypeError("literal_values primitive değer taşımalıdır.")
            if type_name != type(value).__name__:
                raise TypeError("literal_values tür etiketi değerle eşleşmelidir.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("literal_values finite olmalıdır.")
        _require_deep_immutable(self.expression_shape, "expression_shape")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.operator,
            self.branch_polarities,
            self.narrowed_domains,
            tuple(
                (type_name, _canonical_primitive(value))
                for type_name, value in self.literal_values
            ),
            self.expression_shape,
        )


@dataclass(frozen=True, slots=True)
class SafeStateEffect:
    kind: str
    attribute_name: str
    value_domain: SafePrimitiveDomain | None = None
    source_kind: str | None = None
    source_name: str | None = field(default=None, repr=False)
    literal_value: int | float | str | bool | None = field(
        default=None,
        repr=False,
    )
    key_domain: SafePrimitiveDomain | None = None
    key_source_kind: str | None = None
    key_source_name: str | None = field(default=None, repr=False)
    expression_shape: tuple[Any, ...] = field(default=(), repr=False)
    guard_context: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.kind, "kind")
        _require_identifier(self.attribute_name, "attribute_name")
        _require_optional_string(self.source_kind, "source_kind")
        _require_optional_identifier(self.source_name, "source_name")
        _require_optional_string(self.key_source_kind, "key_source_kind")
        _require_optional_identifier(self.key_source_name, "key_source_name")
        if self.value_domain is not None and not isinstance(
            self.value_domain,
            SafePrimitiveDomain,
        ):
            raise TypeError("value_domain SafePrimitiveDomain olmalıdır.")
        if self.key_domain is not None and not isinstance(
            self.key_domain,
            SafePrimitiveDomain,
        ):
            raise TypeError("key_domain SafePrimitiveDomain olmalıdır.")
        _require_optional_primitive(self.literal_value, "literal_value")
        _require_deep_immutable(self.expression_shape, "expression_shape")
        _require_deep_immutable(self.guard_context, "guard_context")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.attribute_name,
            self.value_domain.canonical_payload if self.value_domain else (),
            self.source_kind,
            self.source_name,
            _canonical_primitive(self.literal_value),
            self.key_domain.canonical_payload if self.key_domain else (),
            self.key_source_kind,
            self.key_source_name,
            self.expression_shape,
            self.guard_context,
        )


@dataclass(frozen=True, slots=True)
class SafeLocalCallSummary:
    method_identity: str
    argument_domains: tuple[SafePrimitiveDomain, ...]
    call_depth: int
    callee_summary: SafeMethodSummary = field(repr=False, compare=False)
    callee_execution_fingerprint: str = field(init=False, repr=False)
    callee_semantic_shape_digest: str = field(init=False, repr=False)
    argument_shapes: tuple[tuple[Any, ...], ...] = field(default=(), repr=False)
    guard_context: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.method_identity, "method_identity")
        _require_positive_int(self.call_depth, "call_depth")
        if not isinstance(self.callee_summary, SafeMethodSummary):
            raise TypeError("callee_summary SafeMethodSummary olmalıdır.")
        if not self.callee_summary.supported:
            raise ValueError("callee_summary supported method proof taşımalıdır.")
        if self.callee_summary.receiver.method_identity != self.method_identity:
            raise ValueError("Local call ve callee method identity eşleşmelidir.")
        expected_call_depth = (
            1 + self.callee_summary.budget_footprint.local_call_depth
        )
        if self.call_depth != expected_call_depth:
            raise ValueError(
                "call_depth callee summary transitive depth değerinden türetilmelidir."
            )
        object.__setattr__(
            self,
            "callee_execution_fingerprint",
            self.callee_summary.execution_fingerprint,
        )
        object.__setattr__(
            self,
            "callee_semantic_shape_digest",
            self.callee_summary.semantic_shape_digest,
        )
        _require_sha256(
            self.callee_execution_fingerprint,
            "callee_execution_fingerprint",
        )
        _require_sha256(
            self.callee_semantic_shape_digest,
            "callee_semantic_shape_digest",
        )
        _require_tuple(self.argument_domains, "argument_domains")
        if not all(
            isinstance(item, SafePrimitiveDomain)
            for item in self.argument_domains
        ):
            raise TypeError("argument_domains primitive domain tuple olmalıdır.")
        _require_deep_immutable(self.argument_shapes, "argument_shapes")
        _require_deep_immutable(self.guard_context, "guard_context")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.method_identity,
            tuple(domain.canonical_payload for domain in self.argument_domains),
            self.call_depth,
            self.callee_execution_fingerprint,
            self.argument_shapes,
            self.guard_context,
        )

    @property
    def canonical_semantic_payload(self) -> tuple[Any, ...]:
        return (
            tuple(domain.canonical_payload for domain in self.argument_domains),
            self.call_depth,
            self.callee_semantic_shape_digest,
            self.argument_shapes,
            self.guard_context,
        )


@dataclass(frozen=True, slots=True)
class SafeReturnSummary:
    kind: str
    domain: SafePrimitiveDomain | None = None
    operator: str | None = None
    attribute_name: str | None = field(default=None, repr=False)
    parameter_name: str | None = field(default=None, repr=False)
    local_method_identity: str | None = field(default=None, repr=False)
    literal_values: tuple[tuple[str, int | float | str | bool | None], ...] = field(
        default=(),
        repr=False,
    )
    expression_shape: tuple[Any, ...] = field(default=(), repr=False)
    bounded_cardinality_required: int | None = None
    source_collection_attribute: str | None = field(default=None, repr=False)
    element_primitive_fields: tuple[tuple[str, SafePrimitiveDomain], ...] = field(
        default=(),
        repr=False,
    )
    aggregate_operator: str | None = None
    aggregate_initial_value: int | float | str | bool | None = field(
        default=None,
        repr=False,
    )
    aggregate_filter: SafeGuardSummary | None = field(default=None, repr=False)
    guard_context: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.kind, "kind")
        _require_optional_string(self.operator, "operator")
        _require_optional_identifier(self.attribute_name, "attribute_name")
        _require_optional_identifier(self.parameter_name, "parameter_name")
        if self.domain is not None and not isinstance(
            self.domain,
            SafePrimitiveDomain,
        ):
            raise TypeError("domain SafePrimitiveDomain olmalıdır.")
        _require_deep_immutable(self.literal_values, "literal_values")
        _require_deep_immutable(self.expression_shape, "expression_shape")
        _require_tuple(self.element_primitive_fields, "element_primitive_fields")
        if not all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and isinstance(item[1], SafePrimitiveDomain)
            for item in self.element_primitive_fields
        ):
            raise TypeError("element_primitive_fields canonical tuple olmalıdır.")
        _require_deep_immutable(self.guard_context, "guard_context")
        if self.local_method_identity is not None and (
            type(self.local_method_identity) is not str
            or not self.local_method_identity.isidentifier()
        ):
            raise ValueError("local_method_identity geçerli identifier olmalıdır.")
        if self.bounded_cardinality_required is not None:
            _require_positive_int(
                self.bounded_cardinality_required,
                "bounded_cardinality_required",
            )
        _require_optional_identifier(
            self.source_collection_attribute,
            "source_collection_attribute",
        )
        _require_optional_string(self.aggregate_operator, "aggregate_operator")
        _require_optional_primitive(
            self.aggregate_initial_value,
            "aggregate_initial_value",
        )
        if self.aggregate_filter is not None and not isinstance(
            self.aggregate_filter,
            SafeGuardSummary,
        ):
            raise TypeError("aggregate_filter SafeGuardSummary olmalıdır.")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.domain.canonical_payload if self.domain else (),
            self.operator,
            self.attribute_name,
            self.parameter_name,
            self.local_method_identity,
            tuple(
                (type_name, _canonical_primitive(value))
                for type_name, value in self.literal_values
            ),
            self.expression_shape,
            self.bounded_cardinality_required,
            self.source_collection_attribute,
            tuple(
                (name, domain.canonical_payload)
                for name, domain in self.element_primitive_fields
            ),
            self.aggregate_operator,
            _canonical_primitive(self.aggregate_initial_value),
            (
                self.aggregate_filter.canonical_execution_payload
                if self.aggregate_filter
                else ()
            ),
            self.guard_context,
        )


@dataclass(frozen=True, slots=True)
class SafeMethodBudgetFootprint:
    """Fingerprint'e bağlı, transitive setup maliyetlerinin immutable özeti."""

    state_effects: int
    state_domain_types: tuple[tuple[str, tuple[str, ...]], ...]
    collection_effects: tuple[tuple[str, int], ...]
    observer_calls: int
    mutator_calls: tuple[tuple[str, int], ...]
    local_call_depth: int
    executable_statements: int
    closure_complete: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("state_effects", self.state_effects),
            ("observer_calls", self.observer_calls),
            ("local_call_depth", self.local_call_depth),
            ("executable_statements", self.executable_statements),
        ):
            _require_nonnegative_int(value, name)
        _require_tuple(self.state_domain_types, "state_domain_types")
        _require_tuple(self.collection_effects, "collection_effects")
        _require_tuple(self.mutator_calls, "mutator_calls")
        if type(self.closure_complete) is not bool:
            raise TypeError("closure_complete bool olmalıdır.")
        if any(
            type(name) is not str
            or not name.isidentifier()
            or type(types) is not tuple
            or not all(type(type_name) is str for type_name in types)
            for name, types in self.state_domain_types
        ):
            raise TypeError("state_domain_types canonical tuple olmalıdır.")
        for values, field_name in (
            (self.collection_effects, "collection_effects"),
            (self.mutator_calls, "mutator_calls"),
        ):
            if any(
                type(name) is not str
                or not name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for name, count in values
            ):
                raise TypeError(f"{field_name} canonical count tuple olmalıdır.")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.state_effects,
            self.state_domain_types,
            self.collection_effects,
            self.observer_calls,
            self.mutator_calls,
            self.local_call_depth,
            self.executable_statements,
            self.closure_complete,
        )

    @property
    def canonical_semantic_payload(self) -> tuple[Any, ...]:
        return (
            self.state_effects,
            tuple(types for _, types in self.state_domain_types),
            tuple(count for _, count in self.collection_effects),
            self.observer_calls,
            tuple(count for _, count in self.mutator_calls),
            self.local_call_depth,
            self.executable_statements,
            self.closure_complete,
        )


def _derive_budget_footprint(
    *,
    receiver: SafeReceiverProof,
    guards: tuple[SafeGuardSummary, ...],
    state_effects: tuple[SafeStateEffect, ...],
    local_calls: tuple[SafeLocalCallSummary, ...],
    statement_count: int,
    observer_call_count: int,
    budgets: SafeMethodBudgets,
) -> SafeMethodBudgetFootprint:
    """Analyzer-bound direct ve transitive maliyet closure'ını türetir."""
    state_domain_types: dict[str, set[str]] = {}
    effect_occurrences = [
        (effect.guard_context, 1) for effect in state_effects
    ]
    collection_occurrences: dict[
        str,
        list[tuple[tuple[tuple[int, bool], ...], int]],
    ] = {}
    mutator_occurrences: dict[
        str,
        list[tuple[tuple[tuple[int, bool], ...], int]],
    ] = {}
    statement_occurrences: list[
        tuple[tuple[tuple[int, bool], ...], int]
    ] = []
    observer_occurrences: list[
        tuple[tuple[tuple[int, bool], ...], int]
    ] = []

    if state_effects:
        mutator_occurrences[receiver.method_identity] = [((), 1)]
    for effect in state_effects:
        if (
            effect.kind
            in {
                "ASSIGN_LITERAL",
                "ASSIGN_PARAMETER",
                "AUGMENT_ADD",
                "AUGMENT_SUBTRACT",
            }
            and effect.value_domain is not None
        ):
            state_domain_types.setdefault(
                effect.attribute_name,
                set(),
            ).update(effect.value_domain.admissible_types)
        if effect.kind in {"LIST_APPEND", "DICT_SET"}:
            collection_occurrences.setdefault(
                effect.attribute_name,
                [],
            ).append((effect.guard_context, 1))

    closure_complete = len(guards) <= budgets.guards
    if closure_complete:
        for call in local_calls:
            footprint = call.callee_summary.budget_footprint
            closure_complete = closure_complete and footprint.closure_complete
            if footprint.state_effects:
                effect_occurrences.append(
                    (call.guard_context, footprint.state_effects)
                )
            for attribute_name, count in footprint.collection_effects:
                collection_occurrences.setdefault(
                    attribute_name,
                    [],
                ).append((call.guard_context, count))
            for method_identity, count in footprint.mutator_calls:
                mutator_occurrences.setdefault(
                    method_identity,
                    [],
                ).append((call.guard_context, count))
            for attribute_name, type_names in footprint.state_domain_types:
                state_domain_types.setdefault(
                    attribute_name,
                    set(),
                ).update(type_names)
            statement_occurrences.append(
                (call.guard_context, footprint.executable_statements)
            )
            if footprint.observer_calls:
                observer_occurrences.append(
                    (call.guard_context, footprint.observer_calls)
                )
            elif not footprint.state_effects:
                observer_occurrences.append((call.guard_context, 1))

    if not closure_complete:
        return SafeMethodBudgetFootprint(
            state_effects=len(state_effects),
            state_domain_types=tuple(
                (name, _ordered_types(types))
                for name, types in sorted(state_domain_types.items())
            ),
            collection_effects=tuple(
                (name, sum(count for _, count in occurrences))
                for name, occurrences in sorted(collection_occurrences.items())
            ),
            observer_calls=observer_call_count,
            mutator_calls=(
                ((receiver.method_identity, 1),) if state_effects else ()
            ),
            local_call_depth=max(
                (call.call_depth for call in local_calls),
                default=0,
            ),
            executable_statements=statement_count,
            closure_complete=False,
        )

    transitive_observer_calls = _maximum_guarded_weight(
        len(guards),
        tuple(observer_occurrences),
    )
    if observer_call_count < transitive_observer_calls:
        raise ValueError(
            "observer_call_count callee summary transitive observer "
            "maliyetini eksik sayamaz."
        )

    return SafeMethodBudgetFootprint(
        state_effects=_maximum_guarded_weight(
            len(guards),
            tuple(effect_occurrences),
        ),
        state_domain_types=tuple(
            (name, _ordered_types(types))
            for name, types in sorted(state_domain_types.items())
        ),
        collection_effects=tuple(
            (
                name,
                _maximum_guarded_weight(
                    len(guards),
                    tuple(occurrences),
                ),
            )
            for name, occurrences in sorted(collection_occurrences.items())
        ),
        observer_calls=observer_call_count,
        mutator_calls=tuple(
            (
                method_identity,
                _maximum_guarded_weight(
                    len(guards),
                    tuple(occurrences),
                ),
            )
            for method_identity, occurrences in sorted(
                mutator_occurrences.items()
            )
        ),
        local_call_depth=max(
            (call.call_depth for call in local_calls),
            default=0,
        ),
        executable_statements=(
            statement_count
            + _maximum_guarded_weight(
                len(guards),
                tuple(statement_occurrences),
            )
        ),
        closure_complete=True,
    )


@dataclass(frozen=True, slots=True)
class SafeMethodSummary:
    receiver: SafeReceiverProof
    signature: tuple[tuple[str, SafePrimitiveDomain], ...]
    category: SafeMethodCategory
    guards: tuple[SafeGuardSummary, ...]
    state_effects: tuple[SafeStateEffect, ...]
    local_calls: tuple[SafeLocalCallSummary, ...]
    return_summary: SafeReturnSummary
    operation_kinds: tuple[str, ...]
    budgets: SafeMethodBudgets
    rejection_category: SafeMethodRejectionCategory | None = None
    statement_count: int = 0
    observer_call_count: int = 0
    budget_footprint: SafeMethodBudgetFootprint | None = field(
        default=None,
        repr=False,
    )
    summary_version: int = SUMMARY_VERSION
    _class_state_order: tuple[str, ...] = field(default=(), repr=False)
    _local_order: tuple[str, ...] = field(default=(), repr=False)
    execution_fingerprint: str = field(init=False)
    semantic_shape_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.summary_version != SUMMARY_VERSION:
            raise ValueError("Method summary version desteklenmiyor.")
        if not isinstance(self.category, SafeMethodCategory):
            raise TypeError("category SafeMethodCategory olmalıdır.")
        if self.rejection_category is not None and not isinstance(
            self.rejection_category,
            SafeMethodRejectionCategory,
        ):
            raise TypeError("rejection_category enum olmalıdır.")
        _require_nonnegative_int(self.statement_count, "statement_count")
        _require_nonnegative_int(self.observer_call_count, "observer_call_count")
        _require_tuple(self.signature, "signature")
        _require_tuple(self.guards, "guards")
        _require_tuple(self.state_effects, "state_effects")
        _require_tuple(self.local_calls, "local_calls")
        _require_tuple(self.operation_kinds, "operation_kinds")
        _require_tuple(self._class_state_order, "_class_state_order")
        _require_tuple(self._local_order, "_local_order")
        if not isinstance(self.receiver, SafeReceiverProof):
            raise TypeError("receiver SafeReceiverProof olmalıdır.")
        if not isinstance(self.return_summary, SafeReturnSummary):
            raise TypeError("return_summary SafeReturnSummary olmalıdır.")
        if not isinstance(self.budgets, SafeMethodBudgets):
            raise TypeError("budgets SafeMethodBudgets olmalıdır.")
        if self.budget_footprint is not None and not isinstance(
            self.budget_footprint,
            SafeMethodBudgetFootprint,
        ):
            raise TypeError("budget_footprint SafeMethodBudgetFootprint olmalıdır.")
        if not all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and isinstance(item[1], SafePrimitiveDomain)
            for item in self.signature
        ):
            raise TypeError("signature canonical primitive-domain tuple olmalıdır.")
        if not all(isinstance(item, SafeGuardSummary) for item in self.guards):
            raise TypeError("guards SafeGuardSummary tuple olmalıdır.")
        if not all(isinstance(item, SafeStateEffect) for item in self.state_effects):
            raise TypeError("state_effects SafeStateEffect tuple olmalıdır.")
        if not all(isinstance(item, SafeLocalCallSummary) for item in self.local_calls):
            raise TypeError("local_calls SafeLocalCallSummary tuple olmalıdır.")
        derived_budget_footprint = _derive_budget_footprint(
            receiver=self.receiver,
            guards=self.guards,
            state_effects=self.state_effects,
            local_calls=self.local_calls,
            statement_count=self.statement_count,
            observer_call_count=self.observer_call_count,
            budgets=self.budgets,
        )
        if (
            self.budget_footprint is not None
            and self.budget_footprint != derived_budget_footprint
        ):
            raise ValueError(
                "budget_footprint analyzer-derived transitive closure ile "
                "eşleşmelidir."
            )
        object.__setattr__(
            self,
            "budget_footprint",
            derived_budget_footprint,
        )
        _require_deep_immutable(self.operation_kinds, "operation_kinds")
        _require_deep_immutable(self._class_state_order, "_class_state_order")
        _require_deep_immutable(self._local_order, "_local_order")
        object.__setattr__(
            self,
            "execution_fingerprint",
            _fingerprint(self.canonical_execution_payload),
        )
        object.__setattr__(
            self,
            "semantic_shape_digest",
            _fingerprint(self.canonical_semantic_payload),
        )

    @property
    def qualified_name(self) -> str:
        return f"{self.receiver.class_identity}.{self.receiver.method_identity}"

    @property
    def supported(self) -> bool:
        return (
            self.category is not SafeMethodCategory.UNSUPPORTED
            and self.rejection_category is None
        )

    @property
    def parameter_domains(self) -> tuple[tuple[str, SafePrimitiveDomain], ...]:
        return self.signature

    @property
    def internal_rejection_detail(self) -> None:
        return None

    def parameter_domain(self, parameter_name: str) -> SafePrimitiveDomain:
        for name, domain in self.signature:
            if name == parameter_name:
                return domain
        raise ValueError(f"Primitive parameter domain bulunamadı: {parameter_name}")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.summary_version,
            self.receiver.canonical_execution_payload,
            tuple((name, domain.canonical_payload) for name, domain in self.signature),
            self.category.value,
            tuple(item.canonical_execution_payload for item in self.guards),
            tuple(item.canonical_execution_payload for item in self.state_effects),
            tuple(item.canonical_execution_payload for item in self.local_calls),
            self.return_summary.canonical_execution_payload,
            self.operation_kinds,
            self.budgets.canonical_payload,
            self.rejection_category.value if self.rejection_category else None,
            self.statement_count,
            self.observer_call_count,
            self.budget_footprint.canonical_execution_payload,
        )

    @property
    def canonical_semantic_payload(self) -> tuple[Any, ...]:
        parameter_ordinals = {
            name: f"p{index}" for index, (name, _) in enumerate(self.signature)
        }
        attribute_ordinals = {
            name: f"a{index}" for index, name in enumerate(self._class_state_order)
        }
        local_ordinals = {
            name: f"l{index}" for index, name in enumerate(self._local_order)
        }
        return (
            self.summary_version,
            self.receiver.canonical_semantic_payload,
            tuple(domain.canonical_payload for _, domain in self.signature),
            self.category.value,
            tuple(
                _semantic_guard(item, parameter_ordinals, attribute_ordinals)
                for item in self.guards
            ),
            tuple(
                _semantic_effect(
                    item,
                    parameter_ordinals,
                    attribute_ordinals,
                    local_ordinals,
                )
                for item in self.state_effects
            ),
            tuple(
                _semantic_local_call(
                    item,
                    parameter_ordinals,
                    attribute_ordinals,
                    local_ordinals,
                )
                for item in self.local_calls
            ),
            _semantic_return(
                self.return_summary,
                parameter_ordinals,
                attribute_ordinals,
                local_ordinals,
            ),
            self.operation_kinds,
            self.budgets.canonical_payload,
            self.rejection_category.value if self.rejection_category else None,
            self.statement_count,
            self.observer_call_count,
            self.budget_footprint.canonical_semantic_payload,
        )


@dataclass(frozen=True, slots=True)
class SafeMethodSummaryBundle:
    module_identity: str
    summaries: tuple[SafeMethodSummary, ...]
    summary_version: int = SUMMARY_VERSION

    def __post_init__(self) -> None:
        _validate_module_identity(self.module_identity)
        _require_tuple(self.summaries, "summaries")
        if not all(isinstance(item, SafeMethodSummary) for item in self.summaries):
            raise TypeError("summaries SafeMethodSummary tuple olmalıdır.")
        if any(
            item.receiver.module_identity != self.module_identity
            for item in self.summaries
        ):
            raise ValueError("Bundle ve summary module identity eşleşmelidir.")
        qualified_names = tuple(item.qualified_name for item in self.summaries)
        if len(set(qualified_names)) != len(qualified_names):
            raise ValueError("Bundle qualified method identity tekil olmalıdır.")
        if self.summary_version != SUMMARY_VERSION:
            raise ValueError("Method summary bundle version desteklenmiyor.")

    def summary_for(self, qualified_name: str) -> SafeMethodSummary:
        for summary in self.summaries:
            if summary.qualified_name == qualified_name:
                return summary
        raise ValueError(f"Method summary bulunamadı: {qualified_name}")

    def summaries_for_class(self, class_name: str) -> tuple[SafeMethodSummary, ...]:
        return tuple(
            summary
            for summary in self.summaries
            if summary.receiver.class_identity == class_name
        )


@dataclass(frozen=True, slots=True)
class _StateProof:
    name: str
    kind: str
    domain: SafePrimitiveDomain | None
    literal_value: int | float | str | bool | None = None
    source_parameter_name: str | None = None

    @property
    def execution_payload(self) -> tuple[Any, ...]:
        return (
            self.name,
            self.kind,
            self.domain.canonical_payload if self.domain else (),
            _canonical_primitive(self.literal_value),
            self.source_parameter_name,
        )

    def semantic_payload(
        self,
        parameter_ordinals: dict[str, str],
    ) -> tuple[Any, ...]:
        return (
            self.kind,
            self.domain.canonical_payload if self.domain else (),
            _canonical_primitive(self.literal_value),
            (
                parameter_ordinals.get(self.source_parameter_name, "unknown")
                if self.source_parameter_name is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class _ConstructorProof:
    states: tuple[_StateProof, ...]
    execution_fingerprint: str
    semantic_digest: str
    rejection: SafeMethodRejectionCategory | None = None


@dataclass(frozen=True, slots=True)
class _PendingLocalCall:
    method_name: str
    argument_domains: tuple[SafePrimitiveDomain, ...]
    argument_shapes: tuple[tuple[Any, ...], ...]
    guard_context: tuple[tuple[int, bool], ...]


@dataclass(slots=True)
class _ScanResult:
    domains: dict[str, SafePrimitiveDomain]
    guards: list[SafeGuardSummary] = field(default_factory=list)
    effects: list[SafeStateEffect] = field(default_factory=list)
    pending_calls: list[_PendingLocalCall] = field(default_factory=list)
    returns: list[SafeReturnSummary] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    local_domains: dict[str, SafePrimitiveDomain] = field(default_factory=dict)
    local_expression_shapes: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    local_guard_contexts: dict[str, tuple[tuple[int, bool], ...]] = field(
        default_factory=dict
    )
    local_order: list[str] = field(default_factory=list)
    rejection: SafeMethodRejectionCategory | None = None
    statement_count: int = 0
    observer_call_occurrences: list[
        tuple[tuple[tuple[int, bool], ...], int]
    ] = field(default_factory=list)

    def reject(self, category: SafeMethodRejectionCategory) -> None:
        if self.rejection is None:
            self.rejection = category


class _ClassSummaryAnalyzer:
    def __init__(
        self,
        class_node: ast.ClassDef,
        *,
        module_identity: str,
        budgets: SafeMethodBudgets,
        module_bound_names: frozenset[str],
        module_class_binding_count: int,
        module_namespace_unsafe: bool,
        available_annotation_names: frozenset[str],
    ) -> None:
        self.class_node = class_node
        self.module_identity = module_identity
        self.budgets = budgets
        self.module_bound_names = module_bound_names
        self.methods = {
            statement.name: statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name != "__init__"
        }
        self.constructor = _constructor_proof(class_node)
        self.class_rejection = _class_rejection(
            class_node,
            module_bound_names,
            available_annotation_names,
        )
        self.module_rejection = (
            SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
            if module_class_binding_count != 1 or module_namespace_unsafe
            else None
        )
        self.cache: dict[str, SafeMethodSummary] = {}
        self.budget_footprint_cache: dict[str, SafeMethodBudgetFootprint] = {}

    def analyze(self, method_name: str, stack: tuple[str, ...] = ()) -> SafeMethodSummary:
        cached = self.cache.get(method_name)
        if cached is not None:
            return cached
        method = self.methods[method_name]
        summary = self._analyze_uncached(method, (*stack, method_name))
        self.cache[method_name] = summary
        return summary

    def _budget_footprint(
        self,
        summary: SafeMethodSummary,
    ) -> SafeMethodBudgetFootprint:
        method_name = summary.receiver.method_identity
        cached = self.budget_footprint_cache.get(method_name)
        if cached is not None:
            return cached
        if len(summary.guards) > summary.budgets.guards:
            footprint = replace(
                summary.budget_footprint,
                closure_complete=False,
            )
            self.budget_footprint_cache[method_name] = footprint
            return footprint

        effect_occurrences = [
            (effect.guard_context, 1) for effect in summary.state_effects
        ]
        collection_occurrences: dict[
            str,
            list[tuple[tuple[tuple[int, bool], ...], int]],
        ] = {}
        for effect in summary.state_effects:
            if effect.kind in {"LIST_APPEND", "DICT_SET"}:
                collection_occurrences.setdefault(effect.attribute_name, []).append(
                    (effect.guard_context, 1)
                )
        mutator_occurrences: dict[
            str,
            list[tuple[tuple[tuple[int, bool], ...], int]],
        ] = {}
        if summary.state_effects:
            mutator_occurrences[method_name] = [((), 1)]
        state_domain_types: dict[str, set[str]] = {}
        statement_occurrences: list[
            tuple[tuple[tuple[int, bool], ...], int]
        ] = []
        for effect in summary.state_effects:
            if effect.kind not in {
                "ASSIGN_LITERAL",
                "ASSIGN_PARAMETER",
                "AUGMENT_ADD",
                "AUGMENT_SUBTRACT",
            } or effect.value_domain is None:
                continue
            state_domain_types.setdefault(effect.attribute_name, set()).update(
                effect.value_domain.admissible_types
            )

        for call in summary.local_calls:
            callee = self.cache.get(call.method_identity)
            if callee is None:
                continue
            footprint = self._budget_footprint(callee)
            if footprint.state_effects:
                effect_occurrences.append(
                    (call.guard_context, footprint.state_effects)
                )
            for attribute_name, count in footprint.collection_effects:
                collection_occurrences.setdefault(attribute_name, []).append(
                    (call.guard_context, count)
                )
            for callee_method_name, count in footprint.mutator_calls:
                mutator_occurrences.setdefault(callee_method_name, []).append(
                    (call.guard_context, count)
                )
            for attribute_name, type_names in footprint.state_domain_types:
                state_domain_types.setdefault(attribute_name, set()).update(
                    type_names
                )
            statement_occurrences.append(
                (call.guard_context, footprint.executable_statements)
            )

        footprint = SafeMethodBudgetFootprint(
            state_effects=_maximum_guarded_weight(
                len(summary.guards),
                tuple(effect_occurrences),
            ),
            state_domain_types=tuple(
                (attribute_name, _ordered_types(type_names))
                for attribute_name, type_names in sorted(
                    state_domain_types.items()
                )
            ),
            collection_effects=tuple(
                (
                    attribute_name,
                    _maximum_guarded_weight(
                        len(summary.guards),
                        tuple(occurrences),
                    ),
                )
                for attribute_name, occurrences in sorted(
                    collection_occurrences.items()
                )
            ),
            observer_calls=summary.observer_call_count,
            mutator_calls=tuple(
                (
                    callee_method_name,
                    _maximum_guarded_weight(
                        len(summary.guards),
                        tuple(occurrences),
                    ),
                )
                for callee_method_name, occurrences in sorted(
                    mutator_occurrences.items()
                )
            ),
            local_call_depth=max(
                (call.call_depth for call in summary.local_calls),
                default=0,
            ),
            executable_statements=(
                summary.statement_count
                + _maximum_guarded_weight(
                    len(summary.guards),
                    tuple(statement_occurrences),
                )
            ),
            closure_complete=True,
        )
        self.budget_footprint_cache[method_name] = footprint
        return footprint

    def _analyze_uncached(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        stack: tuple[str, ...],
    ) -> SafeMethodSummary:
        rejection = (
            self.class_rejection
            or self.constructor.rejection
            or self.module_rejection
        )
        if isinstance(method, ast.AsyncFunctionDef):
            rejection = SafeMethodRejectionCategory.ASYNC_METHOD
        elif method.decorator_list:
            if any(
                isinstance(decorator, ast.Name)
                and decorator.id in {"staticmethod", "classmethod"}
                for decorator in method.decorator_list
            ):
                rejection = SafeMethodRejectionCategory.STATIC_OR_CLASS_METHOD
            else:
                rejection = SafeMethodRejectionCategory.DECORATED_METHOD

        receiver_is_self = bool(method.args.args) and method.args.args[0].arg == "self"
        if not receiver_is_self and rejection is None:
            rejection = SafeMethodRejectionCategory.UNSAFE_RECEIVER
        if (
            method.args.posonlyargs
            or method.args.kwonlyargs
            or method.args.vararg is not None
            or method.args.kwarg is not None
        ) and rejection in (None, SafeMethodRejectionCategory.UNSAFE_RECEIVER):
            rejection = SafeMethodRejectionCategory.UNSAFE_SIGNATURE

        parameter_nodes = tuple(method.args.posonlyargs) + tuple(method.args.args)
        if receiver_is_self:
            parameter_nodes = parameter_nodes[1:]
        parameter_names = tuple(argument.arg for argument in parameter_nodes)
        domains, domain_rejection = _parameter_domains(
            method,
            parameter_nodes,
            self.constructor.states,
        )
        if rejection is None:
            rejection = domain_rejection
        has_unproven_parameter = any(
            name not in domains for name in parameter_names
        )

        scan = _MethodScanner(
            method=method,
            class_node=self.class_node,
            states=self.constructor.states,
            domains=domains,
            budgets=self.budgets,
            module_bound_names=self.module_bound_names,
        ).scan()
        if rejection is None:
            rejection = scan.rejection
        if rejection is None and has_unproven_parameter:
            rejection = SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
        guards_within_budget = len(scan.guards) <= self.budgets.guards
        if not guards_within_budget and rejection is None:
            rejection = SafeMethodRejectionCategory.BUDGET_EXCEEDED
        if (
            rejection is None
            and any(item.kind != "NONE" for item in scan.returns)
            and not _block_guarantees_return(method.body)
        ):
            rejection = SafeMethodRejectionCategory.UNSAFE_CONSTRUCT

        local_calls: list[SafeLocalCallSummary] = []
        local_return_summaries: dict[str, SafeReturnSummary] = {}
        local_observer_occurrences: list[tuple[tuple[tuple[int, bool], ...], int]] = []
        effectful_local_call_occurrences: list[
            tuple[str, tuple[tuple[int, bool], ...]]
        ] = []
        transitive_effect_occurrences: list[
            tuple[tuple[tuple[int, bool], ...], int]
        ] = []
        transitive_collection_occurrences: dict[
            str,
            list[tuple[tuple[tuple[int, bool], ...], int]],
        ] = {}
        transitive_mutator_occurrences: dict[
            str,
            list[tuple[tuple[tuple[int, bool], ...], int]],
        ] = {}
        transitive_statement_occurrences: list[
            tuple[tuple[tuple[int, bool], ...], int]
        ] = []
        for pending in scan.pending_calls:
            if pending.method_name in stack:
                if rejection is None:
                    rejection = SafeMethodRejectionCategory.CALL_CYCLE
                continue
            if pending.method_name not in self.methods:
                if rejection is None:
                    rejection = SafeMethodRejectionCategory.UNKNOWN_METHOD
                continue
            callee = self.analyze(pending.method_name, stack)
            if not callee.supported:
                if rejection is None:
                    rejection = (
                        SafeMethodRejectionCategory.CALL_CYCLE
                        if callee.rejection_category
                        is SafeMethodRejectionCategory.CALL_CYCLE
                        else SafeMethodRejectionCategory.UNSAFE_CALL
                    )
                continue
            callee_domains = tuple(domain for _, domain in callee.signature)
            if (
                len(callee_domains) != len(pending.argument_domains)
                or any(
                    not set(actual.admissible_types).issubset(
                        expected.admissible_types
                    )
                    for expected, actual in zip(
                        callee_domains,
                        pending.argument_domains,
                        strict=True,
                    )
                )
            ):
                if rejection is None:
                    rejection = SafeMethodRejectionCategory.DOMAIN_CONFLICT
                continue
            depth = 1 + max(
                (call.call_depth for call in callee.local_calls),
                default=0,
            )
            callee_footprint = self._budget_footprint(callee)
            local_calls.append(
                SafeLocalCallSummary(
                    method_identity=pending.method_name,
                    argument_domains=pending.argument_domains,
                    call_depth=depth,
                    callee_summary=callee,
                    argument_shapes=pending.argument_shapes,
                    guard_context=pending.guard_context,
                )
            )
            local_return_summaries[pending.method_name] = callee.return_summary
            transitive_statement_occurrences.append(
                (pending.guard_context, callee_footprint.executable_statements)
            )
            constructor_domains = {
                state.name: state.domain
                for state in self.constructor.states
                if state.domain is not None
            }
            if rejection is None and any(
                attribute_name not in constructor_domains
                or not set(type_names).issubset(
                    constructor_domains[attribute_name].admissible_types
                )
                for attribute_name, type_names in callee_footprint.state_domain_types
            ):
                rejection = SafeMethodRejectionCategory.UNSAFE_CALL
            if callee_footprint.state_effects:
                effectful_local_call_occurrences.append(
                    (pending.method_name, pending.guard_context)
                )
                if guards_within_budget:
                    transitive_effect_occurrences.append(
                        (
                            pending.guard_context,
                            callee_footprint.state_effects,
                        )
                    )
                    for attribute_name, count in callee_footprint.collection_effects:
                        transitive_collection_occurrences.setdefault(
                            attribute_name,
                            [],
                        ).append(
                            (
                                pending.guard_context,
                                count,
                            )
                        )
                    for callee_method_name, count in callee_footprint.mutator_calls:
                        transitive_mutator_occurrences.setdefault(
                            callee_method_name,
                            [],
                        ).append((pending.guard_context, count))
            if callee_footprint.observer_calls:
                local_observer_occurrences.append(
                    (
                        pending.guard_context,
                        callee_footprint.observer_calls,
                    )
                )
            elif not callee_footprint.state_effects:
                local_observer_occurrences.append((pending.guard_context, 1))
            if depth > self.budgets.local_call_depth and rejection is None:
                rejection = SafeMethodRejectionCategory.BUDGET_EXCEEDED

        collection_effect_occurrences: dict[
            str,
            list[tuple[tuple[tuple[int, bool], ...], int]],
        ] = {}
        for effect in scan.effects:
            if effect.kind not in {"LIST_APPEND", "DICT_SET"}:
                continue
            collection_effect_occurrences.setdefault(
                effect.attribute_name,
                [],
            ).append((effect.guard_context, 1))
        for attribute_name, occurrences in transitive_collection_occurrences.items():
            collection_effect_occurrences.setdefault(
                attribute_name,
                [],
            ).extend(occurrences)
        direct_effect_occurrences = tuple(
            (effect.guard_context, 1) for effect in scan.effects
        )
        if guards_within_budget:
            maximum_effect_count = _maximum_guarded_weight(
                len(scan.guards),
                (
                    *direct_effect_occurrences,
                    *transitive_effect_occurrences,
                )
            )
            maximum_observer_calls = _maximum_guarded_weight(
                len(scan.guards),
                tuple(
                    (
                        *scan.observer_call_occurrences,
                        *local_observer_occurrences,
                    )
                ),
            )
            maximum_setup_calls = _maximum_guarded_weight(
                len(scan.guards),
                tuple(
                    (guard_context, 1)
                    for _, guard_context in effectful_local_call_occurrences
                ),
            )
            repeated_effectful_call_maximums = {
                method_name: _maximum_guarded_weight(
                    len(scan.guards),
                    tuple(occurrences),
                )
                for method_name, occurrences in transitive_mutator_occurrences.items()
            }
            maximum_statement_count = (
                scan.statement_count
                + _maximum_guarded_weight(
                    len(scan.guards),
                    tuple(transitive_statement_occurrences),
                )
            )
        else:
            maximum_effect_count = 0
            maximum_observer_calls = 0
            maximum_setup_calls = 0
            repeated_effectful_call_maximums = {}
            maximum_statement_count = scan.statement_count

        if (
            len(scan.guards) > self.budgets.guards
            or maximum_effect_count > self.budgets.state_effects
            or maximum_observer_calls > self.budgets.observer_calls
            or maximum_statement_count > self.budgets.executable_statements
            or any(
                count > self.budgets.collection_cardinality
                for count in (
                    _maximum_guarded_weight(
                        len(scan.guards),
                        tuple(occurrences),
                    )
                    for occurrences in collection_effect_occurrences.values()
                )
            )
            or maximum_setup_calls > self.budgets.top_level_setup_calls
            or any(
                count > self.budgets.repeated_mutator_calls
                for count in repeated_effectful_call_maximums.values()
            )
        ) and rejection is None:
            rejection = SafeMethodRejectionCategory.BUDGET_EXCEEDED

        resolved_returns: list[SafeReturnSummary] = []
        for item in scan.returns:
            if item.kind != "LOCAL_CALL_RESULT":
                resolved_returns.append(item)
                continue
            callee_return = local_return_summaries.get(
                item.local_method_identity or ""
            )
            if callee_return is None:
                if rejection is None:
                    rejection = SafeMethodRejectionCategory.UNSAFE_CALL
                resolved_returns.append(item)
                continue
            if callee_return.kind == "NONE":
                resolved_returns.append(replace(item, kind="NONE"))
                continue
            if callee_return.domain is None:
                if rejection is None:
                    rejection = SafeMethodRejectionCategory.UNSAFE_CALL
                resolved_returns.append(item)
                continue
            resolved_returns.append(
                replace(
                    item,
                    domain=callee_return.domain,
                    literal_values=callee_return.literal_values,
                )
            )
        scan.returns = resolved_returns

        has_none_return = any(item.kind == "NONE" for item in scan.returns)
        has_value_return = any(item.kind != "NONE" for item in scan.returns)
        if has_none_return and has_value_return and rejection is None:
            rejection = SafeMethodRejectionCategory.UNSAFE_CONSTRUCT

        return_summary = _merge_returns(scan.returns)
        if rejection is not None:
            category = SafeMethodCategory.UNSUPPORTED
        elif local_calls:
            category = SafeMethodCategory.COMPOSED
        elif scan.effects:
            category = (
                SafeMethodCategory.COMPOSED
                if scan.guards or return_summary.kind != "NONE"
                else SafeMethodCategory.MUTATOR
            )
        else:
            category = SafeMethodCategory.OBSERVER

        class_fingerprint = self.constructor.execution_fingerprint
        class_semantic_digest = self.constructor.semantic_digest
        namespace_rejections = tuple(
            item.value
            for item in (self.class_rejection, self.module_rejection)
            if item is not None
        )
        if namespace_rejections:
            class_fingerprint = _fingerprint(
                (class_fingerprint, namespace_rejections)
            )
            class_semantic_digest = _fingerprint(
                (class_semantic_digest, namespace_rejections)
            )
        receiver = SafeReceiverProof(
            module_identity=self.module_identity,
            class_identity=self.class_node.name,
            method_identity=method.name,
            class_fingerprint=class_fingerprint,
            class_semantic_digest=class_semantic_digest,
            direct_self=receiver_is_self,
            class_namespace_safe=(
                self.class_rejection is None and self.module_rejection is None
            ),
            constructor_safe=self.constructor.rejection is None,
        )
        signature = tuple(
            (name, domains[name])
            for name in parameter_names
            if name in domains
        )
        summary = SafeMethodSummary(
            receiver=receiver,
            signature=signature,
            category=category,
            guards=tuple(scan.guards),
            state_effects=tuple(scan.effects),
            local_calls=tuple(local_calls),
            return_summary=return_summary,
            operation_kinds=tuple(scan.operations),
            budgets=self.budgets,
            rejection_category=rejection,
            statement_count=scan.statement_count,
            observer_call_count=maximum_observer_calls,
            _class_state_order=tuple(state.name for state in self.constructor.states),
            _local_order=tuple(scan.local_order),
        )
        return summary


class _MethodScanner:
    def __init__(
        self,
        *,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        class_node: ast.ClassDef,
        states: tuple[_StateProof, ...],
        domains: dict[str, SafePrimitiveDomain],
        budgets: SafeMethodBudgets,
        module_bound_names: frozenset[str],
    ) -> None:
        self.method = method
        self.class_node = class_node
        self.state_by_name = {state.name: state for state in states}
        self.current_state_domains = {
            state.name: state.domain for state in states if state.domain is not None
        }
        self.budgets = budgets
        self.module_bound_names = module_bound_names
        self.scope_bound_names = _bound_names_in_scope(method.body)
        self.result = _ScanResult(domains=dict(domains))
        self.guard_context: tuple[tuple[int, bool], ...] = ()
        self.class_method_names = {
            statement.name
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        positional = (*method.args.posonlyargs, *method.args.args)
        self.method_parameter_names = {
            argument.arg
            for argument in positional
            if argument.arg != "self"
        }

    def scan(self) -> _ScanResult:
        if isinstance(self.method, ast.AsyncFunctionDef):
            self.result.reject(SafeMethodRejectionCategory.ASYNC_METHOD)
        if _contains_yield(self.method):
            self.result.reject(SafeMethodRejectionCategory.GENERATOR)
        if _expression_budget_exceeded(
            self.method,
            max_nodes=self.budgets.executable_statements * 8,
            max_depth=self.budgets.executable_statements * 2,
        ):
            self.result.reject(SafeMethodRejectionCategory.BUDGET_EXCEEDED)
        self._scan_block(self.method.body)
        return self.result

    def _scan_block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._statement(statement)
            if isinstance(statement, (ast.Return, ast.Raise)) or (
                isinstance(statement, ast.If)
                and statement.orelse
                and _block_guarantees_return(statement.body)
                and _block_guarantees_return(statement.orelse)
            ):
                break

    def _statement(self, statement: ast.stmt) -> None:
        self.result.statement_count += 1
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.result.reject(SafeMethodRejectionCategory.NESTED_SCOPE_MUTATION)
            return
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            self.result.reject(SafeMethodRejectionCategory.IMPORT_OR_PROCESS)
            return
        if isinstance(statement, (ast.Global, ast.Nonlocal)):
            self.result.reject(SafeMethodRejectionCategory.GLOBAL_MUTATION)
            return
        if isinstance(statement, ast.Delete):
            self.result.reject(SafeMethodRejectionCategory.DELETE)
            return
        if isinstance(statement, ast.Try):
            self.result.reject(
                SafeMethodRejectionCategory.EXCEPTION_DEPENDENT_EFFECT
            )
            for nested in (*statement.body, *statement.orelse, *statement.finalbody):
                self._statement(nested)
            for handler in statement.handlers:
                for nested in handler.body:
                    self._statement(nested)
            return
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            self.result.reject(SafeMethodRejectionCategory.CONTEXT_MANAGER)
            return
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            self.result.reject(SafeMethodRejectionCategory.UNBOUNDED_CONTROL_FLOW)
            return
        if isinstance(statement, ast.Assign):
            self._assign(statement)
            return
        if isinstance(statement, ast.AnnAssign):
            self._ann_assign(statement)
            return
        if isinstance(statement, ast.AugAssign):
            self._aug_assign(statement)
            return
        if isinstance(statement, ast.Expr):
            if isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value,
                str,
            ):
                return
            if isinstance(statement.value, ast.Call):
                self._expression(statement.value)
                return
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if isinstance(statement, ast.Return):
            result = replace(
                self._return(statement.value),
                guard_context=self.guard_context,
            )
            self.result.returns.append(result)
            self.result.operations.append("RETURN")
            return
        if isinstance(statement, ast.If):
            guard_index = len(self.result.guards)
            self.result.guards.append(self._guard(statement.test))
            self.result.operations.append("GUARD")
            outer_context = self.guard_context
            body_terminates = _block_guarantees_return(statement.body)
            else_terminates = bool(statement.orelse) and _block_guarantees_return(
                statement.orelse
            )
            body_context = (*outer_context, (guard_index, True))
            self.guard_context = body_context
            self._scan_block(statement.body)
            body_fallthrough_context = self.guard_context
            else_context = (*outer_context, (guard_index, False))
            self.guard_context = else_context
            self._scan_block(statement.orelse)
            else_fallthrough_context = self.guard_context
            if (
                not body_terminates
                and body_fallthrough_context != body_context
            ) or (
                not else_terminates
                and else_fallthrough_context != else_context
            ):
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            if body_terminates and not else_terminates:
                self.guard_context = else_context
            elif else_terminates and not body_terminates:
                self.guard_context = body_context
            else:
                self.guard_context = outer_context
            return
        if isinstance(statement, ast.Pass):
            return
        if isinstance(statement, (ast.Break, ast.Continue)):
            self.result.reject(
                SafeMethodRejectionCategory.UNBOUNDED_CONTROL_FLOW
            )
            return
        if isinstance(statement, (ast.Raise, ast.Assert, ast.Match)):
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)

    def _assign(self, statement: ast.Assign) -> None:
        if len(statement.targets) != 1:
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_RECEIVER)
            return
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            if target.id == "self":
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_RECEIVER)
                return
            if target.id in self.method_parameter_names:
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
                return
            if target.id in self.result.local_domains:
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
                return
            domain = self._expression_domain(statement.value)
            self._expression(statement.value)
            if _contains_self(statement.value) and domain is None:
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_RECEIVER)
                return
            if domain is not None:
                self.result.local_domains[target.id] = domain
                self.result.local_expression_shapes[target.id] = self._shape(
                    statement.value
                )
                self.result.local_guard_contexts[target.id] = self.guard_context
                if target.id not in self.result.local_order:
                    self.result.local_order.append(target.id)
            elif _contains_self(statement.value):
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_RECEIVER)
            else:
                self.result.reject(
                    SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
                )
            return
        if _direct_self_attribute(target):
            self._direct_assign(target.attr, statement.value)
            return
        if isinstance(target, ast.Subscript) and _direct_self_attribute(target.value):
            self._dict_assign(target.value.attr, target.slice, statement.value)
            return
        self.result.reject(
            SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH
            if _contains_self(target)
            else SafeMethodRejectionCategory.UNSAFE_CONSTRUCT
        )

    def _ann_assign(self, statement: ast.AnnAssign) -> None:
        if statement.value is None:
            if not isinstance(statement.target, ast.Name):
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        synthetic = ast.Assign(targets=[statement.target], value=statement.value)
        self._assign(synthetic)

    def _direct_assign(self, attribute_name: str, value: ast.expr) -> None:
        state = self.state_by_name.get(attribute_name)
        if state is None:
            self.result.reject(SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH)
            return
        domain = self._expression_domain(value)
        source_kind, source_name, literal = self._value_source(value)
        self._expression(value)
        if source_kind not in {"LITERAL", "PARAMETER"}:
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if domain is None:
            self.result.reject(
                SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
            )
            return
        current_domain = self.current_state_domains.get(attribute_name)
        if (
            current_domain is None
            or not set(domain.admissible_types).issubset(
                current_domain.admissible_types
            )
        ):
            self.result.reject(SafeMethodRejectionCategory.DOMAIN_CONFLICT)
            return
        self.current_state_domains[attribute_name] = _domain(
            set(current_domain.admissible_types) | set(domain.admissible_types),
            ("ORDERED_STATE_EFFECT",),
        )
        self.result.effects.append(
            SafeStateEffect(
                kind="ASSIGN_LITERAL" if source_kind == "LITERAL" else "ASSIGN_PARAMETER",
                attribute_name=attribute_name,
                value_domain=domain,
                source_kind=source_kind,
                source_name=source_name,
                literal_value=literal,
                expression_shape=self._shape(value),
                guard_context=self.guard_context,
            )
        )
        self.result.operations.append("STATE_EFFECT")

    def _dict_assign(
        self,
        attribute_name: str,
        key: ast.expr,
        value: ast.expr,
    ) -> None:
        state = self.state_by_name.get(attribute_name)
        if state is None or state.kind != "EMPTY_DICT":
            self.result.reject(SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH)
            return
        key_domain = self._expression_domain(key)
        value_domain = self._expression_domain(value)
        key_kind, key_name, _ = self._value_source(key)
        value_kind, value_name, literal = self._value_source(value)
        self._expression(key)
        self._expression(value)
        if key_domain is None or value_domain is None:
            self.result.reject(
                SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
            )
            return
        self.result.effects.append(
            SafeStateEffect(
                kind="DICT_SET",
                attribute_name=attribute_name,
                value_domain=value_domain,
                source_kind=value_kind,
                source_name=value_name,
                literal_value=literal,
                key_domain=key_domain,
                key_source_kind=key_kind,
                key_source_name=key_name,
                expression_shape=(
                    "DICT_SET",
                    self._shape(key),
                    self._shape(value),
                ),
                guard_context=self.guard_context,
            )
        )
        self.result.operations.append("STATE_EFFECT")

    def _aug_assign(self, statement: ast.AugAssign) -> None:
        if not _direct_self_attribute(statement.target):
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_RECEIVER)
            return
        state = self.state_by_name.get(statement.target.attr)
        state_domain = self.current_state_domains.get(statement.target.attr)
        domain = self._expression_domain(statement.value)
        source_kind, source_name, literal = self._value_source(statement.value)
        self._expression(statement.value)
        if (
            state is None
            or state_domain is None
            or not set(state_domain.admissible_types).issubset(_NUMERIC_TYPES)
            or domain is None
            or not set(domain.admissible_types).issubset(_NUMERIC_TYPES)
        ):
            self.result.reject(SafeMethodRejectionCategory.DOMAIN_CONFLICT)
            return
        kind = (
            "AUGMENT_ADD"
            if isinstance(statement.op, ast.Add)
            else "AUGMENT_SUBTRACT"
            if isinstance(statement.op, ast.Sub)
            else None
        )
        if kind is None:
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        resulting_types = (
            set(state_domain.admissible_types)
            | set(domain.admissible_types)
        ) & _NUMERIC_TYPES
        self.current_state_domains[statement.target.attr] = _domain(
            tuple(resulting_types),
            ("ORDERED_STATE_EFFECT",),
        )
        self.result.effects.append(
            SafeStateEffect(
                kind=kind,
                attribute_name=statement.target.attr,
                value_domain=domain,
                source_kind=source_kind,
                source_name=source_name,
                literal_value=literal,
                expression_shape=self._shape(statement.value),
                guard_context=self.guard_context,
            )
        )
        self.result.operations.append("STATE_EFFECT")

    def _expression(self, expression: ast.expr) -> None:
        if isinstance(expression, ast.Constant):
            if isinstance(expression.value, float) and not math.isfinite(
                expression.value
            ):
                self.result.reject(SafeMethodRejectionCategory.DOMAIN_CONFLICT)
                return
            if expression.value is None or type(expression.value) in (
                int,
                float,
                str,
                bool,
            ):
                return
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if isinstance(expression, ast.Name):
            if expression.id == "self":
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_RECEIVER)
            elif (
                expression.id not in self.result.domains
                and not self._local_is_available(expression.id)
            ):
                self.result.reject(
                    SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
                )
            return
        if isinstance(expression, ast.Call):
            self._call(expression)
            return
        if isinstance(expression, ast.IfExp):
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if isinstance(expression, ast.Starred):
            self.result.reject(
                SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
            )
            return
        if isinstance(expression, ast.BoolOp) and any(
            isinstance(node, ast.Call) for node in ast.walk(expression)
        ):
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if isinstance(expression, ast.BoolOp):
            for value in expression.values:
                self._expression(value)
            return
        if isinstance(expression, ast.UnaryOp):
            if isinstance(expression.op, ast.Not):
                self._expression(expression.operand)
                return
            if isinstance(expression.op, (ast.UAdd, ast.USub)):
                domain = self._expression_domain(expression.operand)
                if domain is None or not set(domain.admissible_types).issubset(
                    _NUMERIC_TYPES
                ):
                    self.result.reject(
                        SafeMethodRejectionCategory.DOMAIN_CONFLICT
                    )
                    return
                self._expression(expression.operand)
                return
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if isinstance(expression, ast.BinOp):
            if self._expression_domain(expression) is None:
                self.result.reject(SafeMethodRejectionCategory.DOMAIN_CONFLICT)
                return
            self._expression(expression.left)
            self._expression(expression.right)
            return
        if isinstance(expression, ast.Attribute):
            if _direct_self_attribute(expression):
                if expression.attr not in self.state_by_name:
                    self.result.reject(
                        SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH
                    )
                return
            self.result.reject(SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH)
            return
        if isinstance(expression, ast.Subscript):
            self.result.reject(SafeMethodRejectionCategory.UNPROVEN_LOOKUP)
            return
        if isinstance(expression, ast.Compare):
            self._validate_comparison(expression)
            if self.result.rejection is not None:
                return
            self._expression(expression.left)
            for comparator in expression.comparators:
                self._expression(comparator)
            return
        if isinstance(expression, (ast.Lambda, ast.NamedExpr)) or isinstance(
            expression,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return
        if isinstance(expression, (ast.Yield, ast.YieldFrom, ast.Await)):
            self.result.reject(SafeMethodRejectionCategory.GENERATOR)
            return
        self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)

    def _call(self, call: ast.Call) -> None:
        function = call.func
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            for node in ast.walk(function)
        ):
            self.result.reject(SafeMethodRejectionCategory.DYNAMIC_IMPORT)
            return
        if isinstance(function, ast.Name):
            if function.id in {"bool", "len"} and (
                function.id in self.scope_bound_names
                or function.id in self.module_bound_names
                or "*" in self.module_bound_names
            ):
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
                return
            if function.id == "__import__":
                self.result.reject(SafeMethodRejectionCategory.DYNAMIC_IMPORT)
                return
            if function.id in {"eval", "exec", "getattr", "setattr", "delattr"}:
                self.result.reject(SafeMethodRejectionCategory.REFLECTION)
                return
            if function.id == "open":
                self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
                return
            if function.id in self.method_parameter_names:
                self.result.reject(SafeMethodRejectionCategory.CALLBACK)
                return
        for argument in call.args:
            self._expression(argument)
        for keyword in call.keywords:
            self._expression(keyword.value)
        if self.result.rejection is not None:
            return
        if isinstance(function, ast.Name):
            if function.id in {"bool", "len"} and len(call.args) == 1 and not call.keywords:
                argument = call.args[0]
                if _direct_self_attribute(argument):
                    state = self.state_by_name.get(argument.attr)
                    if state is None:
                        self.result.reject(
                            SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH
                        )
                    elif function.id == "len" and state.kind not in {
                        "EMPTY_LIST",
                        "EMPTY_DICT",
                        "EMPTY_TUPLE",
                    }:
                        self.result.reject(
                            SafeMethodRejectionCategory.DOMAIN_CONFLICT
                        )
                    else:
                        self.result.observer_call_occurrences.append(
                            (self.guard_context, 1)
                        )
                    return
                if function.id == "bool" and self._expression_domain(argument):
                    self.result.observer_call_occurrences.append(
                        (self.guard_context, 1)
                    )
                    return
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
            return
        if isinstance(function, ast.Attribute):
            if isinstance(function.value, ast.Call) and isinstance(
                function.value.func,
                ast.Name,
            ) and function.value.func.id == "super":
                self.result.reject(SafeMethodRejectionCategory.DYNAMIC_DISPATCH)
                return
            if _direct_self_attribute(function.value):
                state = self.state_by_name.get(function.value.attr)
                if function.attr == "append" and state is not None:
                    self._collection_append(state, call)
                    return
                if function.attr == "add":
                    self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
                    return
            if isinstance(function.value, ast.Name) and function.value.id == "self":
                self._local_call(function.attr, call)
                return
            if isinstance(function.value, ast.Name) and function.value.id in {
                "subprocess",
                "multiprocessing",
                "os",
                "socket",
            }:
                self.result.reject(SafeMethodRejectionCategory.IMPORT_OR_PROCESS)
                return
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
            return
        self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)

    def _collection_append(self, state: _StateProof, call: ast.Call) -> None:
        if state.kind != "EMPTY_LIST" or len(call.args) != 1 or call.keywords:
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
            return
        value = call.args[0]
        domain = self._expression_domain(value)
        source_kind, source_name, literal = self._value_source(value)
        if domain is None:
            self.result.reject(
                SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
            )
            return
        self.result.effects.append(
            SafeStateEffect(
                kind="LIST_APPEND",
                attribute_name=state.name,
                value_domain=domain,
                source_kind=source_kind,
                source_name=source_name,
                literal_value=literal,
                expression_shape=self._shape(value),
                guard_context=self.guard_context,
            )
        )
        self.result.operations.append("STATE_EFFECT")

    def _local_call(self, method_name: str, call: ast.Call) -> None:
        if method_name in self.state_by_name:
            self.result.reject(SafeMethodRejectionCategory.DYNAMIC_DISPATCH)
            return
        if method_name not in self.class_method_names:
            self.result.reject(SafeMethodRejectionCategory.UNKNOWN_METHOD)
            return
        if call.keywords:
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CALL)
            return
        domains: list[SafePrimitiveDomain] = []
        shapes: list[tuple[Any, ...]] = []
        for argument in call.args:
            domain = self._expression_domain(argument)
            if domain is None:
                self.result.reject(
                    SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
                )
                return
            domains.append(domain)
            shapes.append(self._shape(argument))
        self.result.pending_calls.append(
            _PendingLocalCall(
                method_name,
                tuple(domains),
                tuple(shapes),
                self.guard_context,
            )
        )
        self.result.operations.append("LOCAL_CALL")

    def _guard(self, expression: ast.expr) -> SafeGuardSummary:
        if any(
            not (
                isinstance(call.func, ast.Name)
                and call.func.id in {"bool", "len"}
            )
            for call in ast.walk(expression)
            if isinstance(call, ast.Call)
        ):
            self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
            return SafeGuardSummary(
                kind="UNSUPPORTED",
                operator=None,
                branch_polarities=(True, False),
                expression_shape=self._shape(expression),
            )
        self._expression(expression)
        kind = "BOOLEAN_CONTEXT"
        operator: str | None = None
        literals = _literal_values_in_source_order(expression)
        narrowed: dict[str, tuple[str, ...]] = {}
        for node in ast.walk(expression):
            if isinstance(node, ast.Name) and node.id in self.result.domains:
                narrowed.setdefault(
                    f"parameter:{node.id}",
                    self.result.domains[node.id].admissible_types,
                )
                continue
            if _direct_self_attribute(node):
                state = self.state_by_name.get(node.attr)
                if state is None:
                    continue
                if state.domain is not None:
                    narrowed.setdefault(
                        f"state:{node.attr}",
                        state.domain.admissible_types,
                    )
                elif state.kind in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE"}:
                    narrowed.setdefault(f"state:{node.attr}", (state.kind,))
        if isinstance(expression, ast.Compare):
            operators = tuple(
                _comparison_operator(item) for item in expression.ops
            )
            operator = "|".join(operators) if operators else None
            kind = (
                "MEMBERSHIP"
                if expression.ops
                and isinstance(expression.ops[0], (ast.In, ast.NotIn))
                else "PRIMITIVE_COMPARISON"
            )
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                for node in (expression.left, *expression.comparators)
            ):
                kind = "LENGTH_COMPARISON"
        elif _direct_self_attribute(expression):
            state = self.state_by_name.get(expression.attr)
            kind = (
                "COLLECTION_TRUTHINESS"
                if state is not None
                and state.kind in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE"}
                else "BOOLEAN_CONTEXT"
            )
        elif isinstance(expression, ast.BoolOp):
            kind = "BOOLEAN_AND" if isinstance(expression.op, ast.And) else "BOOLEAN_OR"
        elif isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            state = (
                self.state_by_name.get(expression.operand.attr)
                if _direct_self_attribute(expression.operand)
                else None
            )
            kind = (
                "COLLECTION_TRUTHINESS"
                if state is not None
                and state.kind in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE"}
                else "BOOLEAN_CONTEXT"
            )
        return SafeGuardSummary(
            kind=kind,
            operator=operator,
            branch_polarities=(True, False),
            narrowed_domains=tuple(narrowed.items()),
            literal_values=tuple(literals),
            expression_shape=self._shape(expression),
        )

    def _return(self, expression: ast.expr | None) -> SafeReturnSummary:
        if expression is None or (
            isinstance(expression, ast.Constant) and expression.value is None
        ):
            return SafeReturnSummary(kind="NONE")
        literal = _literal(expression)
        if literal is not None:
            return SafeReturnSummary(
                kind="LITERAL",
                domain=_domain((literal[0],), ("RETURN_LITERAL",), literal[1]),
                literal_values=((literal[0], literal[1]),),
                expression_shape=self._shape(expression),
            )
        if _direct_self_attribute(expression):
            state = self.state_by_name.get(expression.attr)
            domain = self.current_state_domains.get(expression.attr)
            if state is None or domain is None:
                self.result.reject(
                    SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH
                )
                return SafeReturnSummary(kind="UNSUPPORTED")
            return SafeReturnSummary(
                kind="ATTRIBUTE_RETURN",
                domain=domain,
                attribute_name=expression.attr,
                expression_shape=self._shape(expression),
            )
        if isinstance(expression, ast.Name):
            domain = self._expression_domain(expression)
            if domain is None:
                self.result.reject(
                    SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
                )
                return SafeReturnSummary(kind="UNSUPPORTED")
            return SafeReturnSummary(
                kind="PRIMITIVE",
                domain=domain,
                parameter_name=expression.id,
                expression_shape=self._shape(expression),
            )
        if isinstance(expression, ast.Call):
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id == "bool"
                and len(expression.args) == 1
                and not expression.keywords
            ):
                self._call(expression)
                if self.result.rejection is not None:
                    return SafeReturnSummary(kind="UNSUPPORTED")
                argument = expression.args[0]
                return SafeReturnSummary(
                    kind=(
                        "COLLECTION_TRUTHINESS"
                        if _direct_self_attribute(argument)
                        and self.state_by_name[argument.attr].kind
                        in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE"}
                        else "BOOLEAN_CONTEXT"
                    ),
                    domain=_exact_domain("bool", "BOOLEAN_CONTEXT"),
                    attribute_name=(
                        argument.attr if _direct_self_attribute(argument) else None
                    ),
                    parameter_name=(
                        argument.id if isinstance(argument, ast.Name) else None
                    ),
                    expression_shape=self._shape(expression),
                )
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id == "len"
                and len(expression.args) == 1
                and _direct_self_attribute(expression.args[0])
            ):
                self._call(expression)
                if self.result.rejection is not None:
                    return SafeReturnSummary(kind="UNSUPPORTED")
                return SafeReturnSummary(
                    kind="LENGTH",
                    domain=_exact_domain("int", "LENGTH"),
                    attribute_name=expression.args[0].attr,
                    expression_shape=self._shape(expression),
                )
            if (
                isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "append"
                and _direct_self_attribute(expression.func.value)
            ):
                self._call(expression)
                if self.result.rejection is not None:
                    return SafeReturnSummary(kind="UNSUPPORTED")
                return SafeReturnSummary(
                    kind="NONE",
                    expression_shape=self._shape(expression),
                )
            if (
                isinstance(expression.func, ast.Attribute)
                and isinstance(expression.func.value, ast.Name)
                and expression.func.value.id == "self"
            ):
                self._call(expression)
                return SafeReturnSummary(
                    kind="LOCAL_CALL_RESULT",
                    local_method_identity=expression.func.attr,
                    expression_shape=self._shape(expression),
                )
            self._call(expression)
            return SafeReturnSummary(kind="UNSUPPORTED")
        if isinstance(expression, ast.Compare):
            self._expression(expression)
            kind = (
                "MEMBERSHIP"
                if expression.ops
                and isinstance(expression.ops[0], (ast.In, ast.NotIn))
                else "PRIMITIVE_COMPARISON"
            )
            return SafeReturnSummary(
                kind=kind,
                domain=_exact_domain("bool", "COMPARISON"),
                operator=_comparison_operator(expression.ops[0]),
                expression_shape=self._shape(expression),
            )
        if isinstance(expression, ast.BinOp):
            domain = self._expression_domain(expression)
            self._expression(expression)
            if domain is None:
                self.result.reject(SafeMethodRejectionCategory.DOMAIN_CONFLICT)
                return SafeReturnSummary(kind="UNSUPPORTED")
            return SafeReturnSummary(
                kind="ARITHMETIC",
                domain=domain,
                operator=type(expression.op).__name__.upper(),
                expression_shape=self._shape(expression),
            )
        if isinstance(expression, ast.Subscript) and _direct_self_attribute(
            expression.value
        ):
            self.result.reject(SafeMethodRejectionCategory.UNPROVEN_LOOKUP)
            return SafeReturnSummary(
                kind="UNSUPPORTED_LOOKUP",
                attribute_name=expression.value.attr,
                expression_shape=self._shape(expression),
            )
        self._expression(expression)
        self.result.reject(SafeMethodRejectionCategory.UNSAFE_CONSTRUCT)
        return SafeReturnSummary(kind="UNSUPPORTED")

    def _expression_domain(self, expression: ast.expr) -> SafePrimitiveDomain | None:
        literal = _literal(expression)
        if literal is not None:
            return _domain((literal[0],), ("LITERAL",), literal[1])
        if isinstance(expression, ast.Name):
            if expression.id in self.result.domains:
                return self.result.domains[expression.id]
            if self._local_is_available(expression.id):
                return self.result.local_domains[expression.id]
            return None
        if _direct_self_attribute(expression):
            return self.current_state_domains.get(expression.attr)
        if isinstance(expression, ast.Compare):
            return _exact_domain("bool", "COMPARISON")
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return _exact_domain("bool", "BOOLEAN_CONTEXT")
        if isinstance(expression, ast.BinOp):
            left = self._expression_domain(expression.left)
            right = self._expression_domain(expression.right)
            if left is None or right is None:
                return None
            left_types = set(left.admissible_types)
            right_types = set(right.admissible_types)
            if (
                isinstance(expression.op, ast.Add)
                and left_types == {"str"}
                and right_types == {"str"}
            ):
                return _domain(("str",), ("STRING_OPERATION",))
            if not (
                left_types
                and right_types
                and left_types.issubset(_NUMERIC_TYPES)
                and right_types.issubset(_NUMERIC_TYPES)
            ):
                return None
            if isinstance(expression.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                right_literal = _literal(expression.right)
                if right_literal is None or right_literal[1] in (0, 0.0):
                    return None
            if isinstance(expression.op, ast.Div):
                return _domain(("float",), ("NUMERIC_OPERATION",))
            if not isinstance(
                expression.op,
                (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod),
            ):
                return None
            result_types = {
                "int" if left_type == right_type == "int" else "float"
                for left_type in left_types
                for right_type in right_types
            }
            if result_types:
                return _domain(result_types, ("NUMERIC_OPERATION",))
            return None
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            if (
                expression.func.id == "bool"
                and len(expression.args) == 1
                and not expression.keywords
                and (
                    self._expression_domain(expression.args[0]) is not None
                    or _direct_self_attribute(expression.args[0])
                )
            ):
                return _exact_domain("bool", "BOOLEAN_CONTEXT")
            if (
                expression.func.id == "len"
                and len(expression.args) == 1
                and not expression.keywords
                and _direct_self_attribute(expression.args[0])
            ):
                state = self.state_by_name.get(expression.args[0].attr)
                if state and state.kind in {
                    "EMPTY_LIST",
                    "EMPTY_DICT",
                    "EMPTY_TUPLE",
                }:
                    return _exact_domain("int", "LENGTH")
        return None

    def _value_source(
        self,
        expression: ast.expr,
    ) -> tuple[str, str | None, int | float | str | bool | None]:
        literal = _literal(expression)
        if literal is not None:
            return "LITERAL", None, literal[1]
        if isinstance(expression, ast.Name):
            if expression.id in self.result.domains:
                return "PARAMETER", expression.id, None
            if self._local_is_available(expression.id):
                return "LOCAL", expression.id, None
        return "EXPRESSION", None, None

    def _local_is_available(self, name: str) -> bool:
        binding_context = self.result.local_guard_contexts.get(name)
        if binding_context is None:
            return False
        return self.guard_context[: len(binding_context)] == binding_context

    def _validate_comparison(self, expression: ast.Compare) -> None:
        operands = (expression.left, *expression.comparators)
        for operator, left, right in zip(
            expression.ops,
            operands[:-1],
            operands[1:],
            strict=True,
        ):
            left_domain = self._expression_domain(left)
            right_domain = self._expression_domain(right)
            if isinstance(operator, (ast.In, ast.NotIn)):
                if _direct_self_attribute(right):
                    state = self.state_by_name.get(right.attr)
                    if state is None or state.kind not in {
                        "EMPTY_LIST",
                        "EMPTY_DICT",
                        "EMPTY_TUPLE",
                    }:
                        self.result.reject(
                            SafeMethodRejectionCategory.DOMAIN_CONFLICT
                        )
                    elif left_domain is None:
                        self.result.reject(
                            SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
                        )
                    continue
                if (
                    left_domain is not None
                    and right_domain is not None
                    and left_domain.admissible_types == ("str",)
                    and right_domain.admissible_types == ("str",)
                ):
                    continue
                self.result.reject(SafeMethodRejectionCategory.DOMAIN_CONFLICT)
                continue
            if left_domain is None or right_domain is None:
                self.result.reject(
                    SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
                )
                continue
            if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                left_types = set(left_domain.admissible_types)
                right_types = set(right_domain.admissible_types)
                compatible = (
                    left_types.issubset(_NUMERIC_TYPES)
                    and right_types.issubset(_NUMERIC_TYPES)
                ) or left_types == right_types == {"str"}
                if not compatible:
                    self.result.reject(
                        SafeMethodRejectionCategory.DOMAIN_CONFLICT
                    )

    def _shape(self, expression: ast.expr) -> tuple[Any, ...]:
        return _expand_local_bindings(
            _expression_shape(expression),
            self.result.local_expression_shapes,
        )


def analyze_safe_method_summaries(
    tree: ast.Module,
    *,
    module_identity: str,
    budgets: SafeMethodBudgets = DEFAULT_SAFE_METHOD_BUDGETS,
) -> SafeMethodSummaryBundle:
    """Builds immutable static method proofs without executing user code."""
    if not isinstance(tree, ast.Module):
        raise TypeError("tree ast.Module olmalıdır.")
    _validate_module_identity(module_identity)
    if not isinstance(budgets, SafeMethodBudgets):
        raise TypeError("budgets SafeMethodBudgets olmalıdır.")

    module_bound_names = _bound_names_in_scope(tree.body)
    module_binding_counts = _binding_counts_in_scope(tree.body)
    module_namespace_unsafe = _module_namespace_is_unsafe(tree)
    analyzers: list[_ClassSummaryAnalyzer] = []
    available_annotation_names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.ClassDef):
            analyzers.append(
                _ClassSummaryAnalyzer(
                    statement,
                    module_identity=module_identity,
                    budgets=budgets,
                    module_bound_names=module_bound_names,
                    module_class_binding_count=module_binding_counts.get(
                        statement.name,
                        0,
                    ),
                    module_namespace_unsafe=module_namespace_unsafe,
                    available_annotation_names=frozenset(
                        available_annotation_names
                    ),
                )
            )
        if isinstance(statement, ast.ClassDef):
            available_annotation_names.add(statement.name)
    summaries: list[SafeMethodSummary] = []
    seen_qualified_names: set[str] = set()
    for analyzer in analyzers:
        for statement in analyzer.class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if statement.name == "__init__":
                continue
            qualified_name = f"{analyzer.class_node.name}.{statement.name}"
            if qualified_name in seen_qualified_names:
                continue
            seen_qualified_names.add(qualified_name)
            summaries.append(analyzer.analyze(statement.name))
    return SafeMethodSummaryBundle(module_identity, tuple(summaries))


def _maximum_guarded_weight(
    guard_count: int,
    occurrences: tuple[tuple[tuple[tuple[int, bool], ...], int], ...],
) -> int:
    """Returns the largest occurrence weight reachable on one guard assignment."""
    maximum = 0
    for mask in range(1 << guard_count):
        weight = sum(
            occurrence_weight
            for context, occurrence_weight in occurrences
            if all(
                bool(mask & (1 << guard_index)) == polarity
                for guard_index, polarity in context
            )
        )
        maximum = max(maximum, weight)
    return maximum


def _constructor_proof(class_node: ast.ClassDef) -> _ConstructorProof:
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
        execution = (SUMMARY_VERSION, class_node.name, ())
        semantic = (SUMMARY_VERSION, ())
        return _ConstructorProof(
            (),
            _fingerprint(execution),
            _fingerprint(semantic),
        )
    if isinstance(constructor, ast.AsyncFunctionDef) or constructor.decorator_list:
        return _rejected_constructor(class_node.name)
    if (
        constructor.args.posonlyargs
        or constructor.args.kwonlyargs
        or constructor.args.vararg is not None
        or constructor.args.kwarg is not None
    ):
        return _rejected_constructor(class_node.name)
    if not constructor.args.args or constructor.args.args[0].arg != "self":
        return _rejected_constructor(class_node.name)
    parameter_nodes = tuple(constructor.args.posonlyargs) + tuple(constructor.args.args[1:])
    domains, rejection = _parameter_domains(
        constructor,
        parameter_nodes,
        (),
        collect_body_evidence=False,
    )
    if rejection is not None or any(
        parameter.arg not in domains for parameter in parameter_nodes
    ):
        return _rejected_constructor(class_node.name)
    defaults = _defaults_by_name(constructor)
    signature_execution: list[tuple[Any, ...]] = []
    signature_semantic: list[tuple[Any, ...]] = []
    for parameter in parameter_nodes:
        if parameter.arg in defaults:
            default = _literal(defaults[parameter.arg])
            if default is None:
                return _rejected_constructor(class_node.name)
            default_payload: tuple[Any, ...] = (
                "DEFAULT",
                default[0],
                _canonical_primitive(default[1]),
            )
        else:
            default_payload = ("REQUIRED",)
        signature_execution.append(
            (
                parameter.arg,
                domains[parameter.arg].canonical_payload,
                default_payload,
            )
        )
        signature_semantic.append(
            (domains[parameter.arg].canonical_payload, default_payload)
        )

    states: list[_StateProof] = []
    seen: set[str] = set()
    method_names = {
        statement.name
        for statement in class_node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name != "__init__"
    }
    for statement in constructor.body:
        if isinstance(statement, ast.Pass):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return _rejected_constructor(class_node.name)
        target = statement.targets[0]
        if (
            not _direct_self_attribute(target)
            or target.attr in seen
            or target.attr in method_names
            or target.attr.startswith("__")
        ):
            return _rejected_constructor(class_node.name)
        seen.add(target.attr)
        value = statement.value
        literal = _literal(value)
        if literal is not None:
            states.append(
                _StateProof(
                    target.attr,
                    "PRIMITIVE",
                    _domain((literal[0],), ("CONSTRUCTOR_LITERAL",), literal[1]),
                    literal[1],
                )
            )
            continue
        if isinstance(value, ast.Name) and value.id in domains:
            if not domains[value.id].is_exact:
                return _rejected_constructor(class_node.name)
            states.append(
                _StateProof(
                    target.attr,
                    "PRIMITIVE",
                    domains[value.id],
                    source_parameter_name=value.id,
                )
            )
            continue
        empty_kind = _empty_collection_kind(value)
        if empty_kind is not None:
            states.append(_StateProof(target.attr, empty_kind, None))
            continue
        return _rejected_constructor(class_node.name)
    execution = (
        SUMMARY_VERSION,
        class_node.name,
        tuple(signature_execution),
        tuple(state.execution_payload for state in states),
    )
    semantic = (
        SUMMARY_VERSION,
        tuple(signature_semantic),
        tuple(
            state.semantic_payload(
                {
                    parameter.arg: f"p{index}"
                    for index, parameter in enumerate(parameter_nodes)
                }
            )
            for state in states
        ),
    )
    return _ConstructorProof(
        tuple(states),
        _fingerprint(execution),
        _fingerprint(semantic),
    )


def _rejected_constructor(class_name: str) -> _ConstructorProof:
    execution = (SUMMARY_VERSION, class_name, "UNSUPPORTED")
    semantic = (SUMMARY_VERSION, "UNSUPPORTED")
    return _ConstructorProof(
        (),
        _fingerprint(execution),
        _fingerprint(semantic),
        SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE,
    )


def _class_rejection(
    class_node: ast.ClassDef,
    module_bound_names: frozenset[str],
    available_annotation_names: frozenset[str],
) -> SafeMethodRejectionCategory | None:
    if class_node.bases or class_node.keywords:
        return SafeMethodRejectionCategory.INHERITANCE_OR_METACLASS
    if class_node.decorator_list:
        return SafeMethodRejectionCategory.DECORATED_METHOD
    if _class_uses_private_identifier(class_node):
        return SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    seen_method_names: set[str] = set()
    dynamic_protocol_names = {
        "__new__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
    }
    for statement in class_node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            class_visible_names = module_bound_names.union(seen_method_names)
            if (
                statement.name in seen_method_names
                or statement.name in dynamic_protocol_names
                or (
                    statement.name != "__init__"
                    and statement.name.startswith("__")
                )
                or statement.decorator_list
                or not _safe_method_signature(
                    statement,
                    frozenset(class_visible_names),
                    available_annotation_names,
                )
            ):
                return SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
            seen_method_names.add(statement.name)
            continue
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        return SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    return None


def _class_uses_private_identifier(class_node: ast.ClassDef) -> bool:
    for node in ast.walk(class_node):
        identifiers: tuple[str, ...] = ()
        if isinstance(node, ast.Name):
            identifiers = (node.id,)
        elif isinstance(node, ast.arg):
            identifiers = (node.arg,)
        elif isinstance(node, ast.Attribute):
            identifiers = (node.attr,)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            identifiers = (node.name,) if node.name != "__init__" else ()
        elif isinstance(node, ast.keyword) and node.arg is not None:
            identifiers = (node.arg,)
        if any(
            identifier.startswith("__") and not identifier.endswith("__")
            for identifier in identifiers
        ):
            return True
    return False


def _safe_method_signature(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    module_bound_names: frozenset[str],
    available_annotation_names: frozenset[str] = frozenset(),
) -> bool:
    if not _signature_names_are_unique(method):
        return False
    annotations: list[ast.expr] = [
        argument.annotation
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
        if argument.annotation is not None
    ]
    if method.args.vararg is not None and method.args.vararg.annotation is not None:
        annotations.append(method.args.vararg.annotation)
    if method.args.kwarg is not None and method.args.kwarg.annotation is not None:
        annotations.append(method.args.kwarg.annotation)
    if method.returns is not None:
        annotations.append(method.returns)
    if any(
        not _safe_annotation_expression(
            item,
            module_bound_names,
            available_annotation_names,
        )
        for item in annotations
    ):
        return False
    defaults = (
        *method.args.defaults,
        *(item for item in method.args.kw_defaults if item is not None),
    )
    return all(_safe_default_expression(item) for item in defaults)


def _signature_names_are_unique(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    names = [
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
    ]
    if method.args.vararg is not None:
        names.append(method.args.vararg.arg)
    if method.args.kwarg is not None:
        names.append(method.args.kwarg.arg)
    return len(set(names)) == len(names)


def _safe_annotation_expression(
    expression: ast.expr,
    module_bound_names: frozenset[str],
    available_annotation_names: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expression, ast.Constant):
        return expression.value is None or type(expression.value) is str
    if isinstance(expression, ast.Name):
        if expression.id in _PRIMITIVE_TYPES:
            return (
                expression.id not in module_bound_names
                and "*" not in module_bound_names
            )
        return expression.id in available_annotation_names
    if isinstance(expression, ast.Subscript) and isinstance(
        expression.value,
        ast.Name,
    ):
        safe_generic_names = {"list", "dict", "tuple", "set", "frozenset"}
        return (
            expression.value.id in safe_generic_names
            and expression.value.id not in module_bound_names
            and "*" not in module_bound_names
            and _safe_annotation_expression(
                expression.slice,
                module_bound_names,
                available_annotation_names,
            )
        )
    if isinstance(expression, ast.Tuple):
        return all(
            _safe_annotation_expression(
                item,
                module_bound_names,
                available_annotation_names,
            )
            for item in expression.elts
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
        return _safe_union_expression(
            expression,
            module_bound_names,
            available_annotation_names,
        )
    return False


def _safe_union_expression(
    expression: ast.BinOp,
    module_bound_names: frozenset[str],
    available_annotation_names: frozenset[str],
) -> bool:
    operands = (expression.left, expression.right)
    return all(
        _safe_union_operand(
            item,
            module_bound_names,
            available_annotation_names,
        )
        for item in operands
    ) and any(
        isinstance(item, (ast.Name, ast.Subscript))
        or (
            isinstance(item, ast.BinOp)
            and isinstance(item.op, ast.BitOr)
            and _safe_union_expression(
                item,
                module_bound_names,
                available_annotation_names,
            )
        )
        for item in operands
    )


def _safe_union_operand(
    expression: ast.expr,
    module_bound_names: frozenset[str],
    available_annotation_names: frozenset[str],
) -> bool:
    if isinstance(expression, ast.Constant):
        return expression.value is None
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
        return _safe_union_expression(
            expression,
            module_bound_names,
            available_annotation_names,
        )
    if isinstance(expression, (ast.Name, ast.Subscript)):
        return _safe_annotation_expression(
            expression,
            module_bound_names,
            available_annotation_names,
        )
    return False


def _safe_default_expression(expression: ast.expr) -> bool:
    return _literal(expression) is not None or (
        isinstance(expression, ast.Constant) and expression.value is None
    )


def _module_namespace_is_unsafe(tree: ast.Module) -> bool:
    module_bound_names = _bound_names_in_scope(tree.body)
    if module_bound_names.intersection(
        {"__name__", "__builtins__", "__annotations__"}
    ):
        return True
    for statement in tree.body:
        available_annotation_names = _available_names_before(
            tree.body,
            statement,
        )
        if _is_import_main_guard(statement):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.decorator_list or not _safe_method_signature(
                statement,
                module_bound_names,
                available_annotation_names,
            ):
                return True
            continue
        if isinstance(statement, ast.ClassDef):
            if statement.bases or statement.keywords or statement.decorator_list:
                return True
            if "__annotations__" in _bound_names_in_scope(statement.body):
                return True
            class_bound_names: set[str] = set()
            for child in statement.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.decorator_list or not _safe_method_signature(
                        child,
                        frozenset(module_bound_names.union(class_bound_names)),
                        available_annotation_names,
                    ):
                        return True
                    class_bound_names.add(child.name)
                    continue
                if isinstance(child, ast.Pass) or (
                    isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and type(child.value.value) is str
                ):
                    continue
                if not _module_statement_is_inert(
                    child,
                    module_bound_names,
                    available_annotation_names,
                ):
                    return True
            continue
        if isinstance(statement, ast.ImportFrom) and any(
            alias.name == "*" for alias in statement.names
        ):
            return True
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if not _module_statement_is_inert(
            statement,
            module_bound_names,
            available_annotation_names,
        ):
            return True
    return False


def _available_names_before(
    statements: list[ast.stmt],
    target: ast.stmt,
) -> frozenset[str]:
    available: set[str] = set()
    for statement in statements:
        if statement is target:
            break
        if isinstance(statement, ast.ClassDef):
            available.add(statement.name)
    return frozenset(available)


def _module_statement_is_inert(
    statement: ast.stmt,
    module_bound_names: frozenset[str],
    available_annotation_names: frozenset[str],
) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Constant) and type(
            statement.value.value
        ) is str
    if isinstance(statement, ast.Assign):
        return all(_plain_name_target(item) for item in statement.targets) and (
            _module_expression_is_inert(statement.value)
        )
    if isinstance(statement, ast.AnnAssign):
        return (
            _plain_name_target(statement.target)
            and _safe_annotation_expression(
                statement.annotation,
                module_bound_names,
                available_annotation_names,
            )
            and (
                statement.value is None
                or _module_expression_is_inert(statement.value)
            )
        )
    return False


def _plain_name_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_plain_name_target(item) for item in node.elts)
    return False


def _module_expression_is_inert(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Constant):
        return type(expression.value) in (int, float, str, bool, bytes) or (
            expression.value is None or expression.value is Ellipsis
        )
    if _literal(expression) is not None:
        return True
    if isinstance(expression, (ast.Tuple, ast.List)):
        return all(_module_expression_is_inert(item) for item in expression.elts)
    if isinstance(expression, ast.Set):
        return all(
            _module_expression_is_hashable_literal(item)
            for item in expression.elts
        )
    if isinstance(expression, ast.Dict):
        return all(
            key is not None
            and _module_expression_is_hashable_literal(key)
            and _module_expression_is_inert(value)
            for key, value in zip(expression.keys, expression.values, strict=True)
        )
    return False


def _module_expression_is_hashable_literal(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Constant):
        return type(expression.value) in (int, float, str, bool, bytes) or (
            expression.value is None or expression.value is Ellipsis
        )
    if _literal(expression) is not None:
        return True
    if isinstance(expression, ast.Tuple):
        return all(
            _module_expression_is_hashable_literal(item)
            for item in expression.elts
        )
    return False


def _is_import_main_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If) or statement.orelse:
        return False
    test = statement.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    pairs = ((test.left, test.comparators[0]), (test.comparators[0], test.left))
    return any(
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
        for left, right in pairs
    )


def _parameter_domains(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_nodes: tuple[ast.arg, ...],
    states: tuple[_StateProof, ...],
    *,
    collect_body_evidence: bool = True,
) -> tuple[dict[str, SafePrimitiveDomain], SafeMethodRejectionCategory | None]:
    names = tuple(node.arg for node in parameter_nodes)
    constraints: dict[str, list[tuple[frozenset[str], str, object | None]]] = {
        name: [] for name in names
    }
    defaults = _defaults_by_name(method)
    unsupported_annotation = False
    unsupported_default = False
    for node in parameter_nodes:
        annotation_type = _annotation_type(node.annotation)
        if annotation_type is not None:
            constraints[node.arg].append(
                (frozenset({annotation_type}), "ANNOTATION", None)
            )
        elif node.annotation is not None:
            unsupported_annotation = True
            continue
        if node.arg in defaults:
            literal = _literal(defaults[node.arg])
            if literal is None:
                unsupported_default = True
                continue
            constraints[node.arg].append(
                (frozenset({literal[0]}), "DEFAULT_LITERAL", literal[1])
            )
    if collect_body_evidence:
        collector = _DomainEvidenceCollector(set(names), states)
        for statement in method.body:
            collector.visit(statement)
        for name, items in collector.constraints.items():
            constraints[name].extend(items)

    domains: dict[str, SafePrimitiveDomain] = {}
    rejection: SafeMethodRejectionCategory | None = (
        SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
        if unsupported_annotation or unsupported_default
        else None
    )
    for name in names:
        items = constraints[name]
        if not items:
            continue
        admissible = set(_PRIMITIVE_TYPES)
        evidence: list[str] = []
        canonical_value: object | None = None
        for allowed, evidence_kind, value in items:
            admissible &= set(allowed)
            evidence.append(evidence_kind)
            if value is not None:
                canonical_value = value
        if not admissible:
            rejection = SafeMethodRejectionCategory.DOMAIN_CONFLICT
            continue
        ordered = _ordered_types(admissible)
        if canonical_value is not None and type(canonical_value).__name__ not in ordered:
            rejection = SafeMethodRejectionCategory.DOMAIN_CONFLICT
            canonical_value = None
        domains[name] = _domain(ordered, tuple(evidence), canonical_value)
    return domains, rejection


class _DomainEvidenceCollector(ast.NodeVisitor):
    def __init__(self, parameter_names: set[str], states: tuple[_StateProof, ...]) -> None:
        self.parameter_names = parameter_names
        self.state_by_name = {state.name: state for state in states}
        self.constraints: dict[
            str,
            list[tuple[frozenset[str], str, object | None]],
        ] = {name: [] for name in parameter_names}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        self._boolean_context(node.test)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self._boolean_context(node.operand)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = (node.left, *node.comparators)
        for operator, left, right in zip(
            node.ops,
            operands[:-1],
            operands[1:],
            strict=True,
        ):
            self._comparison(operator, left, right)
            self._comparison(operator, right, left)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._binary(node.op, node.left, node.right)
        self._binary(node.op, node.right, node.left)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _direct_self_attribute(node.value) and isinstance(node.slice, ast.Name):
            self._add(
                node.slice.id,
                _HASHABLE_PRIMITIVE_TYPES,
                "HASHABLE_REQUIREMENT",
            )
        self.generic_visit(node)

    def _boolean_context(self, node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            self._add(node.id, {"bool"}, "BOOLEAN_CONTEXT")

    def _comparison(
        self,
        operator: ast.cmpop,
        candidate: ast.expr,
        other: ast.expr,
    ) -> None:
        if not isinstance(candidate, ast.Name):
            return
        literal = _literal(other)
        if isinstance(operator, (ast.In, ast.NotIn)) and _direct_self_attribute(other):
            state = self.state_by_name.get(other.attr)
            allowed = (
                _HASHABLE_PRIMITIVE_TYPES
                if state and state.kind == "EMPTY_DICT"
                else _PRIMITIVE_TYPES
            )
            self._add(candidate.id, allowed, "HASHABLE_REQUIREMENT")
            return
        if literal is None:
            return
        type_name, value = literal
        if isinstance(operator, (ast.Is, ast.IsNot)):
            if type_name == "bool":
                self._add(candidate.id, {"bool"}, "BOOLEAN_IDENTITY", value)
            return
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            allowed = _NUMERIC_TYPES if type_name in _NUMERIC_TYPES else {type_name}
            self._add(candidate.id, allowed, "EXACT_LITERAL_COMPARISON", value)
            return
        if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            if type_name == "bool":
                self._add(candidate.id, set(), "INVALID_ORDERED_BOOL")
                return
            allowed = _NUMERIC_TYPES if type_name in _NUMERIC_TYPES else {"str"}
            self._add(candidate.id, allowed, "ORDERED_COMPARISON", value)

    def _binary(
        self,
        operator: ast.operator,
        candidate: ast.expr,
        other: ast.expr,
    ) -> None:
        if not isinstance(candidate, ast.Name):
            return
        literal = _literal(other)
        if isinstance(operator, ast.Add) and literal and literal[0] == "str":
            self._add(candidate.id, {"str"}, "STRING_OPERATION")
            return
        if isinstance(operator, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            if literal is None or literal[0] in _NUMERIC_TYPES:
                self._add(candidate.id, _NUMERIC_TYPES, "NUMERIC_OPERATION")

    def _add(
        self,
        name: str,
        allowed: set[str] | frozenset[str],
        evidence: str,
        value: object | None = None,
    ) -> None:
        if name in self.constraints:
            self.constraints[name].append((frozenset(allowed), evidence, value))


def _merge_returns(returns: list[SafeReturnSummary]) -> SafeReturnSummary:
    if not returns:
        return SafeReturnSummary(kind="NONE")
    if len(returns) == 1:
        return returns[0]
    if all(item == returns[0] for item in returns[1:]):
        return returns[0]
    domain_types: set[str] = set()
    literals: list[tuple[str, int | float | str | bool | None]] = []
    for item in returns:
        if item.domain is not None:
            domain_types.update(item.domain.admissible_types)
        literals.extend(item.literal_values)
    domain = (
        _domain(tuple(domain_types), ("BRANCHED_RETURN",))
        if domain_types
        else None
    )
    return SafeReturnSummary(
        kind="BRANCHED",
        domain=domain,
        literal_values=tuple(literals),
        expression_shape=tuple(
            (item.guard_context, item.expression_shape) for item in returns
        ),
    )


def _semantic_guard(
    item: SafeGuardSummary,
    parameters: dict[str, str],
    attributes: dict[str, str],
) -> tuple[Any, ...]:
    return (
        item.kind,
        item.operator,
        item.branch_polarities,
        tuple(
            (
                _semantic_narrowed_identity(name, parameters, attributes),
                domain,
            )
            for name, domain in item.narrowed_domains
        ),
        tuple(
            (type_name, _canonical_primitive(value))
            for type_name, value in item.literal_values
        ),
        _normalize_shape(item.expression_shape, parameters, attributes, {}),
    )


def _semantic_narrowed_identity(
    name: str,
    parameters: dict[str, str],
    attributes: dict[str, str],
) -> str:
    source_kind, separator, raw_name = name.partition(":")
    if separator and source_kind == "parameter":
        return f"parameter:{parameters.get(raw_name, 'unknown')}"
    if separator and source_kind == "state":
        return f"state:{attributes.get(raw_name, 'unknown')}"
    return parameters.get(name, attributes.get(name, "unknown"))


def _semantic_effect(
    item: SafeStateEffect,
    parameters: dict[str, str],
    attributes: dict[str, str],
    locals_: dict[str, str],
) -> tuple[Any, ...]:
    source = (
        parameters.get(item.source_name, item.source_name)
        if item.source_kind == "PARAMETER"
        else locals_.get(item.source_name, item.source_name)
        if item.source_kind == "LOCAL"
        else None
    )
    key_source = (
        parameters.get(item.key_source_name, item.key_source_name)
        if item.key_source_kind == "PARAMETER"
        else locals_.get(item.key_source_name, item.key_source_name)
        if item.key_source_kind == "LOCAL"
        else None
    )
    return (
        item.kind,
        attributes.get(item.attribute_name, "unknown"),
        item.value_domain.canonical_payload if item.value_domain else (),
        item.source_kind,
        source,
        _canonical_primitive(item.literal_value),
        item.key_domain.canonical_payload if item.key_domain else (),
        item.key_source_kind,
        key_source,
        _normalize_shape(
            item.expression_shape,
            parameters,
            attributes,
            locals_,
        ),
        item.guard_context,
    )


def _semantic_local_call(
    item: SafeLocalCallSummary,
    parameters: dict[str, str],
    attributes: dict[str, str],
    locals_: dict[str, str],
) -> tuple[Any, ...]:
    return (
        tuple(domain.canonical_payload for domain in item.argument_domains),
        item.call_depth,
        item.callee_semantic_shape_digest,
        tuple(
            _normalize_shape(shape, parameters, attributes, locals_)
            for shape in item.argument_shapes
        ),
        item.guard_context,
    )


def _semantic_return(
    item: SafeReturnSummary,
    parameters: dict[str, str],
    attributes: dict[str, str],
    locals_: dict[str, str],
) -> tuple[Any, ...]:
    return (
        item.kind,
        item.domain.canonical_payload if item.domain else (),
        item.operator,
        attributes.get(item.attribute_name, None),
        parameters.get(
            item.parameter_name,
            locals_.get(item.parameter_name, None),
        ),
        item.local_method_identity is not None,
        tuple(
            (type_name, _canonical_primitive(value))
            for type_name, value in item.literal_values
        ),
        _normalize_shape(item.expression_shape, parameters, attributes, locals_),
        item.bounded_cardinality_required,
        attributes.get(item.source_collection_attribute, None),
        tuple(
            (f"f{index}", domain.canonical_payload)
            for index, (_, domain) in enumerate(item.element_primitive_fields)
        ),
        item.aggregate_operator,
        _canonical_primitive(item.aggregate_initial_value),
        (
            _semantic_guard(item.aggregate_filter, parameters, attributes)
            if item.aggregate_filter
            else ()
        ),
        item.guard_context,
    )


def _expression_shape(node: ast.AST) -> tuple[Any, ...]:
    if isinstance(node, ast.Constant):
        literal = _literal(node)
        return ("LITERAL", literal) if literal is not None else ("NONE",)
    if isinstance(node, ast.Name):
        return ("NAME", node.id)
    if _direct_self_attribute(node):
        return ("SELF_ATTRIBUTE", node.attr)
    if isinstance(node, ast.Attribute):
        return ("ATTRIBUTE", _expression_shape(node.value), node.attr)
    if isinstance(node, ast.BinOp):
        return (
            "BINARY",
            type(node.op).__name__.upper(),
            _expression_shape(node.left),
            _expression_shape(node.right),
        )
    if isinstance(node, ast.UnaryOp):
        return (
            "UNARY",
            type(node.op).__name__.upper(),
            _expression_shape(node.operand),
        )
    if isinstance(node, ast.Compare):
        return (
            "COMPARE",
            tuple(_comparison_operator(operator) for operator in node.ops),
            _expression_shape(node.left),
            tuple(_expression_shape(item) for item in node.comparators),
        )
    if isinstance(node, ast.BoolOp):
        return (
            "BOOL",
            type(node.op).__name__.upper(),
            tuple(_expression_shape(item) for item in node.values),
        )
    if isinstance(node, ast.Call):
        return (
            "CALL",
            _expression_shape(node.func),
            tuple(_expression_shape(item) for item in node.args),
        )
    if isinstance(node, ast.Subscript):
        return (
            "SUBSCRIPT",
            _expression_shape(node.value),
            _expression_shape(node.slice),
        )
    return (type(node).__name__.upper(),)


def _normalize_shape(
    shape: tuple[Any, ...],
    parameters: dict[str, str],
    attributes: dict[str, str],
    locals_: dict[str, str],
) -> tuple[Any, ...]:
    if not shape:
        return ()
    if shape[0] == "NAME" and len(shape) == 2:
        name = shape[1]
        return (
            "NAME",
            parameters.get(name, locals_.get(name, "unknown")),
        )
    if shape[0] == "SELF_ATTRIBUTE" and len(shape) == 2:
        return ("SELF_ATTRIBUTE", attributes.get(shape[1], "unknown"))
    normalized: list[Any] = []
    for item in shape:
        if isinstance(item, tuple):
            normalized.append(
                _normalize_shape(item, parameters, attributes, locals_)
            )
        else:
            normalized.append(item)
    return tuple(normalized)


def _expand_local_bindings(
    shape: tuple[Any, ...],
    bindings: dict[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    if not shape:
        return ()
    if shape[0] == "NAME" and len(shape) == 2 and shape[1] in bindings:
        return ("LOCAL_BINDING", bindings[shape[1]])
    return tuple(
        _expand_local_bindings(item, bindings)
        if isinstance(item, tuple)
        else item
        for item in shape
    )


def _expression_budget_exceeded(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    max_nodes: int,
    max_depth: int,
) -> bool:
    stack: list[tuple[ast.AST, int]] = [
        (statement, 0) for statement in reversed(method.body)
    ]
    expression_nodes = 0
    while stack:
        node, depth = stack.pop()
        if isinstance(node, ast.expr):
            expression_nodes += 1
            if expression_nodes > max_nodes or depth > max_depth:
                return True
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        for child in reversed(tuple(ast.iter_child_nodes(node))):
            stack.append((child, depth + 1))
    return False


class _ScopeBindingCollector(ast.NodeVisitor):
    """Collects lexical bindings without crossing a nested scope boundary."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.counts: dict[str, int] = {}

    def _record(self, name: str) -> None:
        self.names.add(name)
        self.counts[name] = self.counts.get(name, 0) + 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._record(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if isinstance(node.name, str):
            self._record(node.name)
        for statement in node.body:
            self.visit(statement)


def _bound_names_in_scope(statements: list[ast.stmt]) -> frozenset[str]:
    collector = _ScopeBindingCollector()
    for statement in statements:
        collector.visit(statement)
    return frozenset(collector.names)


def _binding_counts_in_scope(statements: list[ast.stmt]) -> dict[str, int]:
    collector = _ScopeBindingCollector()
    for statement in statements:
        collector.visit(statement)
    return dict(collector.counts)


def _block_guarantees_return(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if (
            isinstance(statement, ast.If)
            and statement.orelse
            and _block_guarantees_return(statement.body)
            and _block_guarantees_return(statement.orelse)
        ):
            return True
    return False


def _annotation_type(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name) and annotation.id in _PRIMITIVE_TYPES:
        return annotation.id
    return None


def _defaults_by_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, ast.expr]:
    positional = (*node.args.posonlyargs, *node.args.args)
    defaults = {
        argument.arg: default
        for argument, default in zip(
            positional[-len(node.args.defaults) :] if node.args.defaults else (),
            node.args.defaults,
            strict=True,
        )
    }
    defaults.update(
        {
            argument.arg: default
            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
                strict=True,
            )
            if default is not None
        }
    )
    return defaults


def _literal(
    node: ast.AST | None,
) -> tuple[str, int | float | str | bool] | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool):
        if isinstance(node.value, float) and not math.isfinite(node.value):
            return None
        return type(node.value).__name__, node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    ):
        value = +node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return type(value).__name__, value
    return None


def _literal_values_in_source_order(
    node: ast.AST,
) -> list[tuple[str, int | float | str | bool]]:
    values: list[tuple[str, int | float | str | bool]] = []

    def collect(current: ast.AST) -> None:
        literal = _literal(current)
        if literal is not None:
            values.append(literal)
            return
        for child in ast.iter_child_nodes(current):
            collect(child)

    collect(node)
    return values


def _empty_collection_kind(node: ast.expr) -> str | None:
    if isinstance(node, ast.List) and not node.elts:
        return "EMPTY_LIST"
    if isinstance(node, ast.Dict) and not node.keys:
        return "EMPTY_DICT"
    if isinstance(node, ast.Tuple) and not node.elts:
        return "EMPTY_TUPLE"
    return None


def _domain(
    types: tuple[str, ...] | set[str] | frozenset[str],
    evidence: tuple[str, ...],
    canonical_value: object | None = None,
) -> SafePrimitiveDomain:
    if canonical_value is not None and type(canonical_value) not in (int, float, str, bool):
        raise TypeError("Canonical primitive değer desteklenmiyor.")
    return SafePrimitiveDomain(
        admissible_types=_ordered_types(types),
        evidence_kinds=tuple(evidence),
        canonical_value=canonical_value,  # type: ignore[arg-type]
    )


def _exact_domain(type_name: str, evidence: str) -> SafePrimitiveDomain:
    return _domain((type_name,), (evidence,))


def _ordered_types(values: Any) -> tuple[str, ...]:
    value_set = set(values)
    if not value_set.issubset(_PRIMITIVE_TYPES):
        raise ValueError("Bilinmeyen primitive domain türü.")
    return tuple(item for item in _PRIMITIVE_ORDER if item in value_set)


def _require_tuple(value: object, field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} immutable tuple olmalıdır.")


def _require_string(value: object, field_name: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} boş olmayan string olmalıdır.")


def _require_optional_string(value: object, field_name: str) -> None:
    if value is not None:
        _require_string(value, field_name)


def _require_identifier(value: object, field_name: str) -> None:
    _require_string(value, field_name)
    if not value.isidentifier():
        raise ValueError(f"{field_name} geçerli identifier olmalıdır.")


def _require_optional_identifier(value: object, field_name: str) -> None:
    if value is not None:
        _require_identifier(value, field_name)


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise TypeError(f"{field_name} pozitif tam sayı olmalıdır.")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field_name} negatif olmayan tam sayı olmalıdır.")


def _require_sha256(value: object, field_name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} canonical SHA-256 olmalıdır.")


def _require_optional_primitive(value: object, field_name: str) -> None:
    if value is None:
        return
    if type(value) not in (int, float, str, bool):
        raise TypeError(f"{field_name} primitive olmalıdır.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} finite olmalıdır.")


def _require_deep_immutable(value: object, field_name: str) -> None:
    _require_tuple(value, field_name)
    stack = list(value)
    while stack:
        item = stack.pop()
        if type(item) is tuple:
            stack.extend(item)
            continue
        if item is None or type(item) in (str, int, float, bool):
            continue
        raise TypeError(f"{field_name} yalnız immutable canonical değer taşımalıdır.")


def _canonical_primitive(value: object | None) -> tuple[Any, ...]:
    if value is None:
        return ("none",)
    if type(value) not in (int, float, str, bool):
        raise TypeError("Canonical payload yalnız primitive değer taşır.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN ve infinity canonical payload'a giremez.")
    return (type(value).__name__, value)


def _fingerprint(payload: tuple[Any, ...]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _comparison_operator(operator: ast.cmpop) -> str:
    names = {
        ast.Eq: "EQUAL",
        ast.NotEq: "NOT_EQUAL",
        ast.Lt: "LESS_THAN",
        ast.LtE: "LESS_THAN_OR_EQUAL",
        ast.Gt: "GREATER_THAN",
        ast.GtE: "GREATER_THAN_OR_EQUAL",
        ast.Is: "IS",
        ast.IsNot: "IS_NOT",
        ast.In: "IN",
        ast.NotIn: "NOT_IN",
    }
    return names.get(type(operator), type(operator).__name__.upper())


def _direct_self_attribute(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _contains_self(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Name) and item.id == "self" for item in ast.walk(node))


def _contains_yield(node: ast.AST) -> bool:
    for descendant in ast.walk(node):
        if descendant is node:
            continue
        if isinstance(descendant, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(descendant, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def _validate_module_identity(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(not part.isidentifier() for part in value.split("."))
    ):
        raise ValueError("module_identity canonical dotted Python adı olmalıdır.")
