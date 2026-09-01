from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from types import FunctionType, ModuleType
from typing import Any

from analyzer.primitive_parameter_inference import (
    infer_primitive_parameter_types,
    primitive_annotation_type,
    primitive_literal_type,
)
from analyzer.safe_custom_object import (
    UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER,
    SafeCustomObjectTargetSpec,
    SafeObjectConstructionBlueprint,
    SafeObjectParameterSpec,
)
from analyzer.safe_method_summary import (
    DEFAULT_SAFE_METHOD_BUDGETS,
    SafeMethodBudgets,
    SafeGuardSummary,
    SafeLocalCallSummary,
    SafeMethodCategory,
    SafeMethodRejectionCategory,
    SafeMethodSummary,
    SafePrimitiveDomain,
    SafeReturnSummary,
    SafeStateEffect,
    analyze_safe_method_summaries,
)
from analyzer.simple_instance_method import SimpleInstanceMethodSpec
from cfg.path_analyzer import ExecutionPath


_SETUP_PLAN_VERSION = 1
_PRIMITIVE_TYPES: dict[str, type[object]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
}


class SafeSetupPlanRejectionCategory(str, Enum):
    PRIMITIVE_DOMAIN_UNRESOLVED = "PRIMITIVE_DOMAIN_UNRESOLVED"
    CROSS_MODULE_OBJECT = "CROSS_MODULE_OBJECT"
    UNSAFE_OBJECT_GRAPH = "UNSAFE_OBJECT_GRAPH"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    UNSUPPORTED_METHOD_PROOF = "UNSUPPORTED_METHOD_PROOF"
    INVALID_GUARD_SELECTION = "INVALID_GUARD_SELECTION"
    INVALID_RUNTIME_BINDING = "INVALID_RUNTIME_BINDING"


class SafeSetupPlanRejection(ValueError):
    """Bounded setup proof/plan uyuşmazlığını kontrollü biçimde taşır."""

    category: SafeSetupPlanRejectionCategory

    def __init__(
        self,
        category: SafeSetupPlanRejectionCategory,
        message: str,
    ) -> None:
        if not isinstance(category, SafeSetupPlanRejectionCategory):
            raise TypeError("category SafeSetupPlanRejectionCategory olmalıdır.")
        self.category = category
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SafeTypedArgumentBinding:
    """Primitive değer veya aynı plandaki owned object slot referansıdır."""

    parameter_name: str
    type_name: str
    value: int | float | str | bool | None = field(default=None, repr=False)
    object_slot_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_identifier(self.parameter_name, "parameter_name")
        if self.type_name == "object":
            if self.value is not None or self.object_slot_id is None:
                _reject(
                    SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                    "Object argument tam olarak bir slot referansı taşımalıdır.",
                )
            _require_identifier(self.object_slot_id, "object_slot_id")
            return

        expected = _PRIMITIVE_TYPES.get(self.type_name)
        if (
            expected is None
            or self.object_slot_id is not None
            or type(self.value) is not expected
            or isinstance(self.value, float)
            and not math.isfinite(self.value)
        ):
            _reject(
                SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
                "Setup argument exact ve finite primitive domain taşımalıdır.",
            )

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.parameter_name,
            self.type_name,
            _canonical_primitive(self.value),
            self.object_slot_id,
        )


@dataclass(frozen=True, slots=True)
class SafeObjectSlot:
    """Tek bir constructor çağrısının alias-free plan slot'udur."""

    slot_id: str
    blueprint: SafeObjectConstructionBlueprint = field(repr=False)
    relation_depth: int = 0
    owner_slot_id: str | None = field(default=None, repr=False)
    proven_empty_attributes: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _require_identifier(self.slot_id, "slot_id")
        if not isinstance(self.blueprint, SafeObjectConstructionBlueprint):
            raise TypeError("blueprint SafeObjectConstructionBlueprint olmalıdır.")
        if (
            isinstance(self.relation_depth, bool)
            or not isinstance(self.relation_depth, int)
            or self.relation_depth < 0
        ):
            raise ValueError("relation_depth negatif olmayan tam sayı olmalıdır.")
        if self.owner_slot_id is not None:
            _require_identifier(self.owner_slot_id, "owner_slot_id")
        if self.relation_depth == 0 and self.owner_slot_id is not None:
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Root object slot owner taşıyamaz.",
            )
        if self.relation_depth > 0 and self.owner_slot_id is None:
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Nested object slot doğrulanmış owner taşımalıdır.",
            )
        if self.owner_slot_id == self.slot_id:
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Object slot kendisini owner olarak kullanamaz.",
            )
        _require_tuple(self.proven_empty_attributes, "proven_empty_attributes")
        if (
            len(set(self.proven_empty_attributes))
            != len(self.proven_empty_attributes)
            or not all(
                type(name) is str and name.isidentifier()
                for name in self.proven_empty_attributes
            )
        ):
            raise ValueError("proven_empty_attributes tekil identifier olmalıdır.")

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.slot_id,
            self.blueprint.module_identity,
            self.blueprint.class_name,
            self.blueprint.class_fingerprint,
            self.blueprint.depth,
            tuple(
                (
                    name,
                    type(value).__name__,
                    _canonical_primitive(value),
                )
                for name, value in self.blueprint.constructor_arguments
            ),
            self.relation_depth,
            self.owner_slot_id,
            self.proven_empty_attributes,
        )


@dataclass(frozen=True, slots=True)
class SafeSetupCall:
    """D1 proof'una bağlı, source-order gerçek setup method çağrısıdır."""

    receiver_slot_id: str
    method_summary: SafeMethodSummary = field(repr=False)
    arguments: tuple[SafeTypedArgumentBinding, ...] = field(
        default=(),
        repr=False,
    )
    selected_guard_outcomes: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.receiver_slot_id, "receiver_slot_id")
        if not isinstance(self.method_summary, SafeMethodSummary):
            raise TypeError("method_summary SafeMethodSummary olmalıdır.")
        if not self.method_summary.supported:
            _reject(
                SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF,
                "Unsupported method summary setup planına bağlanamaz.",
            )
        if not (
            self.method_summary.receiver.class_namespace_safe
            and self.method_summary.receiver.constructor_safe
        ):
            _reject(
                SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF,
                "Setup receiver güvenli class ve constructor proof taşımalıdır.",
            )
        if not self.method_summary.budget_footprint.closure_complete:
            _reject(
                SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF,
                "Setup method transitive budget closure kanıtı taşımalıdır.",
            )
        _require_tuple(self.arguments, "arguments")
        if not all(
            isinstance(argument, SafeTypedArgumentBinding)
            for argument in self.arguments
        ):
            raise TypeError("arguments typed binding tuple olmalıdır.")
        names = tuple(argument.parameter_name for argument in self.arguments)
        if len(set(names)) != len(names):
            raise ValueError("Setup call argument adları tekrar edemez.")
        _require_tuple(self.selected_guard_outcomes, "selected_guard_outcomes")
        guard_indices: list[int] = []
        for item in self.selected_guard_outcomes:
            if (
                type(item) is not tuple
                or len(item) != 2
                or isinstance(item[0], bool)
                or not isinstance(item[0], int)
                or item[0] < 0
                or type(item[1]) is not bool
            ):
                _reject(
                    SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION,
                    "Guard selection canonical (index, bool) tuple olmalıdır.",
                )
            index, outcome = item
            if (
                index >= len(self.method_summary.guards)
                or outcome
                not in self.method_summary.guards[index].branch_polarities
            ):
                _reject(
                    SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION,
                    "Selected guard outcome method proof'uyla eşleşmiyor.",
                )
            guard_indices.append(index)
        if len(set(guard_indices)) != len(guard_indices):
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION,
                "Aynı guard bir setup çağrısında tekrar seçilemez.",
            )

        selected = dict(self.selected_guard_outcomes)
        required_contexts = (
            *(effect.guard_context for effect in self.method_summary.state_effects),
            *(call.guard_context for call in self.method_summary.local_calls),
            self.method_summary.return_summary.guard_context,
        )
        if any(
            selected.get(index) is not outcome
            for context in required_contexts
            for index, outcome in context
        ):
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION,
                "Selected guard outcome yürütülen proof context'ini kapsamıyor.",
            )

        supplied = {argument.parameter_name: argument for argument in self.arguments}
        for name, domain in self.method_summary.signature:
            argument = supplied.get(name)
            if argument is None:
                _reject(
                    SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
                    "Method proof için gereken primitive setup argument bulunamadı.",
                )
            if (
                argument.type_name == "object"
                or argument.type_name not in domain.admissible_types
            ):
                _reject(
                    SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
                    "Setup argument method proof primitive domain'iyle uyuşmuyor.",
                )

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.receiver_slot_id,
            self.method_summary.execution_fingerprint,
            tuple(item.canonical_execution_payload for item in self.arguments),
            self.selected_guard_outcomes,
        )


@dataclass(frozen=True, slots=True)
class SafeTargetBinding:
    """Target parametresini plandaki materialized object slot'a bağlar."""

    parameter_name: str
    object_slot_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.parameter_name, "parameter_name")
        _require_identifier(self.object_slot_id, "object_slot_id")


@dataclass(frozen=True, slots=True)
class SafeObjectSetupPlan:
    """Public modellere açılmayan bounded proof-to-runtime planıdır."""

    module_identity: str = field(repr=False)
    target_identity: str
    object_slots: tuple[SafeObjectSlot, ...] = field(repr=False)
    setup_calls: tuple[SafeSetupCall, ...] = field(default=(), repr=False)
    target_bindings: tuple[SafeTargetBinding, ...] = field(
        default=(),
        repr=False,
    )
    receiver_slot_id: str | None = field(default=None, repr=False)
    budgets: SafeMethodBudgets = field(
        default=DEFAULT_SAFE_METHOD_BUDGETS,
        repr=False,
    )
    plan_version: int = field(default=_SETUP_PLAN_VERSION, repr=False)
    pre_state_fingerprint: str = field(init=False, repr=False)
    post_state_fingerprint: str = field(init=False, repr=False)
    execution_fingerprint: str = field(init=False, repr=False)
    semantic_shape_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_module_identity(self.module_identity)
        _validate_qualified_identity(self.target_identity)
        if self.plan_version != _SETUP_PLAN_VERSION:
            raise ValueError("Setup plan version desteklenmiyor.")
        if not isinstance(self.budgets, SafeMethodBudgets):
            raise TypeError("budgets SafeMethodBudgets olmalıdır.")
        _require_tuple(self.object_slots, "object_slots")
        _require_tuple(self.setup_calls, "setup_calls")
        _require_tuple(self.target_bindings, "target_bindings")
        if self.receiver_slot_id is not None:
            _require_identifier(self.receiver_slot_id, "receiver_slot_id")
        if not self.object_slots:
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Setup plan en az bir object slot taşımalıdır.",
            )
        if not all(isinstance(item, SafeObjectSlot) for item in self.object_slots):
            raise TypeError("object_slots SafeObjectSlot tuple olmalıdır.")
        if not all(isinstance(item, SafeSetupCall) for item in self.setup_calls):
            raise TypeError("setup_calls SafeSetupCall tuple olmalıdır.")
        if not all(
            isinstance(item, SafeTargetBinding) for item in self.target_bindings
        ):
            raise TypeError("target_bindings SafeTargetBinding tuple olmalıdır.")

        self._validate_objects()
        self._validate_calls()
        self._validate_target_bindings()
        self._validate_budgets()

        pre_payload = (
            self.plan_version,
            tuple(item.canonical_execution_payload for item in self.object_slots),
        )
        post_payload = (
            pre_payload,
            tuple(item.canonical_execution_payload for item in self.setup_calls),
        )
        object.__setattr__(
            self,
            "pre_state_fingerprint",
            _fingerprint(pre_payload),
        )
        object.__setattr__(
            self,
            "post_state_fingerprint",
            _fingerprint(post_payload),
        )
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

    def __repr__(self) -> str:
        return (
            "SafeObjectSetupPlan("
            f"target_identity={self.target_identity!r}, "
            f"object_count={len(self.object_slots)}, "
            f"setup_call_count={len(self.setup_calls)})"
        )

    @property
    def canonical_execution_payload(self) -> tuple[Any, ...]:
        return (
            self.plan_version,
            self.module_identity,
            self.target_identity,
            tuple(item.canonical_execution_payload for item in self.object_slots),
            tuple(item.canonical_execution_payload for item in self.setup_calls),
            tuple(
                (item.parameter_name, item.object_slot_id)
                for item in self.target_bindings
            ),
            self.receiver_slot_id,
            self.budgets.canonical_payload,
        )

    @property
    def canonical_semantic_payload(self) -> tuple[Any, ...]:
        slot_ordinals = {
            slot.slot_id: f"o{index}"
            for index, slot in enumerate(self.object_slots)
        }
        return (
            self.plan_version,
            tuple(
                (
                    slot.blueprint.depth,
                    tuple(
                        (
                            type(value).__name__,
                            _canonical_primitive(value),
                        )
                        for _, value in slot.blueprint.constructor_arguments
                    ),
                    slot.relation_depth,
                    (
                        slot_ordinals[slot.owner_slot_id]
                        if slot.owner_slot_id is not None
                        else None
                    ),
                    tuple("EMPTY_COLLECTION" for _ in slot.proven_empty_attributes),
                )
                for slot in self.object_slots
            ),
            tuple(
                (
                    slot_ordinals[call.receiver_slot_id],
                    call.method_summary.semantic_shape_digest,
                    tuple(
                        (
                            argument.type_name,
                            _canonical_primitive(argument.value),
                            (
                                slot_ordinals[argument.object_slot_id]
                                if argument.object_slot_id is not None
                                else None
                            ),
                        )
                        for argument in call.arguments
                    ),
                    call.selected_guard_outcomes,
                )
                for call in self.setup_calls
            ),
            tuple(
                slot_ordinals[item.object_slot_id]
                for item in self.target_bindings
            ),
            (
                slot_ordinals[self.receiver_slot_id]
                if self.receiver_slot_id is not None
                else None
            ),
            self.budgets.canonical_payload,
        )

    def _validate_objects(self) -> None:
        slot_ids = tuple(item.slot_id for item in self.object_slots)
        if len(set(slot_ids)) != len(slot_ids):
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Object slot identity tekil olmalıdır.",
            )
        slots = {item.slot_id: item for item in self.object_slots}
        for slot in self.object_slots:
            if slot.blueprint.module_identity != self.module_identity:
                _reject(
                    SafeSetupPlanRejectionCategory.CROSS_MODULE_OBJECT,
                    "Object slot yalnız planın canonical local modülünü kullanabilir.",
                )
            if slot.owner_slot_id is None:
                continue
            owner = slots.get(slot.owner_slot_id)
            if owner is None or owner.relation_depth + 1 != slot.relation_depth:
                _reject(
                    SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                    "Nested object ownership zinciri tekil ve depth uyumlu olmalıdır.",
                )

    def _validate_calls(self) -> None:
        slots = {item.slot_id: item for item in self.object_slots}
        for call in self.setup_calls:
            receiver = slots.get(call.receiver_slot_id)
            if receiver is None:
                _reject(
                    SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                    "Setup call receiver slot bulunamadı.",
                )
            assert receiver is not None
            proof = call.method_summary.receiver
            if (
                proof.module_identity != self.module_identity
                or proof.class_identity != receiver.blueprint.class_name
                or proof.class_fingerprint
                != receiver.blueprint.class_fingerprint
                or not proof.direct_self
            ):
                _reject(
                    SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF,
                    "Setup call receiver D1 method proof'uyla eşleşmiyor.",
                )
            selected = dict(call.selected_guard_outcomes)
            for guard_index, expected_outcome in selected.items():
                actual_outcome = _evaluate_setup_guard(
                    call.method_summary.guards[guard_index],
                    call=call,
                    receiver=receiver,
                    slots=slots,
                )
                if actual_outcome is None or actual_outcome is not expected_outcome:
                    _reject(
                        SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION,
                        "Selected guard outcome bound setup witness ile uyuşmuyor.",
                    )
            for argument in call.arguments:
                if argument.object_slot_id is None:
                    continue
                nested = slots.get(argument.object_slot_id)
                if (
                    nested is None
                    or nested.owner_slot_id != receiver.slot_id
                    or nested.relation_depth != receiver.relation_depth + 1
                ):
                    _reject(
                        SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                        "Setup object argument alias-free owned relation olmalıdır.",
                    )
                matching_callees = tuple(
                    local_call.callee_summary
                    for local_call in call.method_summary.local_calls
                    if (
                        local_call.callee_summary.receiver.class_identity
                        == nested.blueprint.class_name
                    )
                )
                if (
                    len(matching_callees) != 1
                    or matching_callees[0].receiver.module_identity
                    != self.module_identity
                    or matching_callees[0].receiver.class_fingerprint
                    != nested.blueprint.class_fingerprint
                    or not matching_callees[0].receiver.class_namespace_safe
                    or not matching_callees[0].receiver.constructor_safe
                ):
                    _reject(
                        SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF,
                        "Nested object slot transitive receiver proof ile eşleşmelidir.",
                    )

    def _validate_target_bindings(self) -> None:
        slots = {item.slot_id for item in self.object_slots}
        names = tuple(item.parameter_name for item in self.target_bindings)
        if len(set(names)) != len(names):
            raise ValueError("Target binding parametre adları tekrar edemez.")
        if any(item.object_slot_id not in slots for item in self.target_bindings):
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Target binding bilinmeyen object slot kullanıyor.",
            )
        if self.receiver_slot_id is not None and self.receiver_slot_id not in slots:
            _reject(
                SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
                "Method receiver binding bilinmeyen object slot kullanıyor.",
            )

    def _validate_budgets(self) -> None:
        if len(self.object_slots) > self.budgets.max_objects:
            _budget_rejection("object", len(self.object_slots))
        if any(
            slot.blueprint.depth > self.budgets.constructor_depth
            for slot in self.object_slots
        ):
            _budget_rejection("constructor depth", self.budgets.constructor_depth + 1)
        relation_depth = max(slot.relation_depth for slot in self.object_slots)
        if relation_depth > self.budgets.setup_relation_depth:
            _budget_rejection("setup relation depth", relation_depth)
        if len(self.setup_calls) > self.budgets.top_level_setup_calls:
            _budget_rejection("top-level setup call", len(self.setup_calls))
        repeated: dict[str, int] = {}
        for call in self.setup_calls:
            for identity, count in call.method_summary.budget_footprint.mutator_calls:
                repeated[identity] = repeated.get(identity, 0) + count
        repeated_count = max(repeated.values(), default=0)
        if repeated_count > self.budgets.repeated_mutator_calls:
            _budget_rejection("repeated mutator call", repeated_count)

        state_effect_count = sum(
            call.method_summary.budget_footprint.state_effects
            for call in self.setup_calls
        )
        if state_effect_count > self.budgets.state_effects:
            _budget_rejection("state effect", state_effect_count)
        collection_cardinality = sum(
            count
            for call in self.setup_calls
            for _, count in call.method_summary.budget_footprint.collection_effects
        )
        if collection_cardinality > self.budgets.collection_cardinality:
            _budget_rejection("collection cardinality", collection_cardinality)
        local_call_depth = max(
            (
                call.method_summary.budget_footprint.local_call_depth
                for call in self.setup_calls
            ),
            default=0,
        )
        if local_call_depth > self.budgets.local_call_depth:
            _budget_rejection("local call depth", local_call_depth)
        guard_count = sum(
            len(call.selected_guard_outcomes) for call in self.setup_calls
        )
        if guard_count > self.budgets.guards:
            _budget_rejection("guard outcome", guard_count)
        observer_count = sum(
            call.method_summary.budget_footprint.observer_calls
            for call in self.setup_calls
        )
        if observer_count > self.budgets.observer_calls:
            _budget_rejection("observer call", observer_count)
        statement_count = sum(
            call.method_summary.budget_footprint.executable_statements
            for call in self.setup_calls
        )
        if statement_count > self.budgets.executable_statements:
            _budget_rejection("executable statement", statement_count)


@dataclass(frozen=True, slots=True)
class _ConstructorArgumentProof:
    name: str
    type_name: str


@dataclass(frozen=True, slots=True)
class _ObserverProof:
    method_name: str
    analysis_name: str
    return_type: str
    value_attribute: str | None = None
    target_argument_types: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CorrelatedCollectionItem:
    """Tek logical collection item'ına ait atomik primitive projection'lar."""

    item_index: int
    collection_key: int | float | str | bool
    projections: tuple[tuple[tuple[str, ...], int | float | str | bool], ...]

    def __post_init__(self) -> None:
        if type(self.item_index) is not int or not 0 <= self.item_index < 2:
            raise ValueError("Correlated item index bounded olmalıdır.")
        if type(self.collection_key) not in _PRIMITIVE_TYPES.values():
            raise TypeError("Correlated collection key primitive olmalıdır.")
        _require_tuple(self.projections, "projections")
        paths = tuple(path for path, _ in self.projections)
        if len(set(paths)) != len(paths):
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Aynı correlated projection çelişkili değer taşıyamaz.",
            )
        for path, value in self.projections:
            if (
                not path
                or any(type(part) is not str or not part for part in path)
                or type(value) not in _PRIMITIVE_TYPES.values()
            ):
                raise TypeError("Correlated projection exact primitive olmalıdır.")

    @property
    def canonical_payload(self) -> tuple[Any, ...]:
        return (
            self.item_index,
            _canonical_primitive(self.collection_key),
            tuple(
                (path, _canonical_primitive(value))
                for path, value in self.projections
            ),
        )


@dataclass(frozen=True, slots=True)
class CorrelatedCollectionWitness:
    """En fazla iki item taşıyan deterministic collection witness'ı."""

    collection_identity: str
    items: tuple[CorrelatedCollectionItem, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.collection_identity, "collection_identity")
        _require_tuple(self.items, "items")
        if len(self.items) > 2:
            _reject(
                SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED,
                "Correlated collection witness iki item bütçesini aşıyor.",
            )
        if not all(isinstance(item, CorrelatedCollectionItem) for item in self.items):
            raise TypeError("items CorrelatedCollectionItem tuple olmalıdır.")

    @property
    def canonical_payload(self) -> tuple[Any, ...]:
        return (
            "CORRELATED_COLLECTION_WITNESS_V1",
            self.collection_identity,
            tuple(item.canonical_payload for item in self.items),
        )


@dataclass(frozen=True, slots=True)
class SafeMethodCollectionSetupContext:
    """Basit instance method collection path'lerini bounded setup'a bağlar."""

    module_identity: str = field(repr=False)
    target_identity: str
    method_spec: SimpleInstanceMethodSpec = field(repr=False)
    collection_attribute: str
    pattern_kind: str
    root_class_fingerprint: str = field(repr=False)
    nested_class_name: str
    nested_constructor: tuple[_ConstructorArgumentProof, ...] = field(repr=False)
    nested_class_fingerprint: str = field(repr=False)
    nested_key_attribute: str
    nested_value_attribute: str
    nested_stock_attribute: str
    setup_object_parameter: str
    setup_quantity_parameter: str
    setup_quantity_type: str
    setup_summary: SafeMethodSummary = field(repr=False)
    item_projection_paths: tuple[tuple[str, ...], ...] = ()
    record_projection_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    selection_operator: str | None = None
    selected_identity_projection: tuple[str, ...] | None = None
    selected_value_projection: tuple[str, ...] | None = None
    aggregate_analysis_name: str | None = None
    iterable_analysis_name: str | None = None
    loop_target_name: str = ""
    accumulator_name: str | None = None
    selected_name: str | None = None
    item_object_name: str | None = None
    state_assignment_label: str = ""
    loop_label: str = ""
    neutral_assignment_labels: tuple[str, ...] = ()
    update_label: str | None = None
    state_condition_label: str | None = None
    initial_selection_condition_label: str | None = None
    selection_condition_label: str | None = None
    initial_condition_analysis_names: tuple[str, ...] = ()
    selection_condition_analysis_names: tuple[str, ...] = ()
    return_label: str = ""
    append_label: str | None = None

    @property
    def maximum_item_count(self) -> int:
        return 2

    def analysis_metadata(
        self,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        names = list(parameter_names)
        types = dict(parameter_types)
        for private_name in (
            self.aggregate_analysis_name,
            self.iterable_analysis_name,
        ):
            if private_name is not None and private_name not in names:
                names.append(private_name)
        if self.aggregate_analysis_name is not None:
            nested_types = {item.name: item.type_name for item in self.nested_constructor}
            types[self.aggregate_analysis_name] = nested_types[self.nested_value_attribute]
        if self.iterable_analysis_name is not None:
            types[self.iterable_analysis_name] = "list"
        for name in (
            *self.initial_condition_analysis_names,
            *self.selection_condition_analysis_names,
        ):
            if name not in names:
                names.append(name)
            types[name] = "bool"
        return tuple(names), types

    def rewrite_path(self, path: ExecutionPath) -> ExecutionPath:
        if not isinstance(path, ExecutionPath):
            raise TypeError("path ExecutionPath olmalıdır.")
        iteration_count = _path_iteration_count(path, self.loop_label)
        if iteration_count > self.maximum_item_count:
            return path
        labels: list[str] = []
        initial_condition_index = 0
        selection_condition_index = 0
        for index, label in enumerate(path.node_labels):
            outgoing = (
                path.edge_labels[index]
                if index < len(path.edge_labels)
                else None
            )
            if label == self.state_assignment_label:
                labels.append("__safe_collection_marker = 0")
            elif label == self.loop_label:
                labels.append(
                    f"{self.loop_target_name} in {self.iterable_analysis_name}"
                )
            elif label in self.neutral_assignment_labels:
                if self.pattern_kind == "ARG_EXTREME" and self.item_object_name:
                    labels.append(
                        f"{self.item_object_name} = {self.loop_target_name}"
                    )
                else:
                    target = ast.parse(label).body[0]
                    if isinstance(target, ast.Assign):
                        labels.append(f"{ast.unparse(target.targets[0])} = 0")
                    else:
                        labels.append(label)
            elif label == self.update_label:
                labels.append(f"{self.accumulator_name} += 0")
            elif (
                self.pattern_kind == "SCALAR_FOLD"
                and self.accumulator_name is not None
                and label == f"{self.accumulator_name} = 0"
            ):
                labels.append(
                    f"{self.accumulator_name} = {self.aggregate_analysis_name}"
                )
            elif label == self.state_condition_label:
                labels.append(f"not {self.iterable_analysis_name}")
            elif label == self.initial_selection_condition_label:
                labels.append(
                    self.initial_condition_analysis_names[
                        initial_condition_index
                    ]
                )
                initial_condition_index += 1
            elif label == self.selection_condition_label:
                labels.append(
                    self.selection_condition_analysis_names[
                        selection_condition_index
                    ]
                )
                selection_condition_index += 1
            elif label == self.return_label and self.pattern_kind == "ARG_EXTREME":
                labels.append(f"return {self.selected_name}")
            elif label == self.append_label:
                labels.append("result.append(__safe_collection_index)")
            else:
                labels.append(label)
        return ExecutionPath(
            node_ids=list(path.node_ids),
            edge_labels=list(path.edge_labels),
            node_labels=labels,
            node_types=list(path.node_types),
            line_numbers=list(path.line_numbers),
        )

    def correlated_witness_for(
        self,
        *,
        item_values: tuple[int | float, ...],
        collection_nonempty: bool,
    ) -> CorrelatedCollectionWitness:
        _require_tuple(item_values, "item_values")
        if type(collection_nonempty) is not bool:
            raise TypeError("collection_nonempty bool olmalıdır.")
        if not collection_nonempty and item_values:
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Boş collection için correlated item constraint çelişkilidir.",
            )
        if len(item_values) > self.maximum_item_count:
            _reject(
                SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED,
                "Correlated item witness iki item bütçesini aşıyor.",
            )
        if any(type(value) not in {int, float} for value in item_values):
            _reject(
                SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
                "Correlated item scalar değeri numeric olmalıdır.",
            )
        key_type = {
            item.name: item.type_name for item in self.nested_constructor
        }[self.nested_key_attribute]
        projection_paths = tuple(dict.fromkeys(
            (
                *(self.item_projection_paths),
                self.selected_value_projection or (self.nested_value_attribute,),
            )
        ))
        items = tuple(
            CorrelatedCollectionItem(
                item_index=index,
                collection_key=_indexed_setup_value(key_type, index),
                projections=tuple(
                    (
                        path,
                        (
                            value
                            if path == (
                                self.selected_value_projection
                                or (self.nested_value_attribute,)
                            )
                            else 1
                        ),
                    )
                    for path in projection_paths
                    if path != ("$key",)
                ),
            )
            for index, value in enumerate(item_values)
        )
        return CorrelatedCollectionWitness(self.collection_attribute, items)

    def bind_generated_input(
        self,
        generated_input: Any,
        *,
        path: ExecutionPath | None = None,
    ) -> Any:
        from generator.path_input_generator import GeneratedTestInput

        if not isinstance(generated_input, GeneratedTestInput):
            raise TypeError("generated_input GeneratedTestInput olmalıdır.")
        if path is None:
            raise TypeError("Correlated setup binding ExecutionPath gerektirir.")
        iteration_count = _path_iteration_count(path, self._rewritten_loop_label())
        if iteration_count > self.maximum_item_count:
            _reject(
                SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED,
                "Correlated path iki item bütçesini aşıyor.",
            )
        values = dict(generated_input.keyword_arguments)
        item_values: tuple[int | float, ...]
        if self.pattern_kind == "SCALAR_FOLD":
            aggregate = values.get(self.aggregate_analysis_name)
            if type(aggregate) not in {int, float}:
                _reject(
                    SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                    "Scalar fold aggregate exact numeric olmalıdır.",
                )
            item_values = (
                (aggregate,) if iteration_count == 1 else (aggregate, 0)
            ) if iteration_count else ()
        elif self.pattern_kind == "ARG_EXTREME":
            raw_values = values.get(self.iterable_analysis_name)
            if type(raw_values) is not list or len(raw_values) != iteration_count:
                _reject(
                    SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                    "Arg-extreme iterable path cardinality ile eşleşmelidir.",
                )
            item_values = self._arg_extreme_values(path, iteration_count)
        else:
            item_values = tuple(index + 1 for index in range(iteration_count))

        witness = self.correlated_witness_for(
            item_values=item_values,
            collection_nonempty=bool(iteration_count),
        )
        plan = self._setup_plan(witness)
        expected_result = generated_input.expected_result
        if self.pattern_kind == "ARG_EXTREME" and witness.items:
            selector = max if self.selection_operator == ">" else min
            selected = selector(
                witness.items,
                key=lambda item: dict(item.projections)[
                    self.selected_value_projection or (self.nested_value_attribute,)
                ],
            )
            expected_result = selected.collection_key
        elif self.pattern_kind == "RECORD_PROJECTION":
            expected_result = [
                {
                    output_name: self._projection_value(item, projection)
                    for output_name, projection in self.record_projection_paths
                }
                for item in witness.items
            ]
        private_names = {
            name for name in (self.aggregate_analysis_name, self.iterable_analysis_name)
            if name is not None
        }
        private_names.update(self.initial_condition_analysis_names)
        private_names.update(self.selection_condition_analysis_names)
        constructor_names = {
            parameter.analysis_name for parameter in self.method_spec.constructor_parameters
        }
        return replace(
            generated_input,
            keyword_arguments=tuple(
                (name, value)
                for name, value in generated_input.keyword_arguments
                if name not in private_names and name not in constructor_names
            ),
            expected_result=expected_result,
            setup_plan=plan,
        )

    def _rewritten_loop_label(self) -> str:
        return f"{self.loop_target_name} in {self.iterable_analysis_name}"

    def _arg_extreme_values(
        self,
        path: ExecutionPath,
        iteration_count: int,
    ) -> tuple[int, ...]:
        initialization = tuple(
            path.edge_labels[index] == "True"
            for index, label in enumerate(path.node_labels[:-1])
            if label in self.initial_condition_analysis_names
        )
        if initialization != tuple(index == 0 for index in range(iteration_count)):
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Arg-extreme selected-item lifecycle path ile çelişiyor.",
            )
        comparisons = tuple(
            path.edge_labels[index] == "True"
            for index, label in enumerate(path.node_labels[:-1])
            if label in self.selection_condition_analysis_names
        )
        if len(comparisons) != max(0, iteration_count - 1):
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Arg-extreme comparison sayısı path cardinality ile çelişiyor.",
            )
        values = [0]
        for selected, comparison_true in enumerate(comparisons):
            previous = values[selected]
            if self.selection_operator == ">":
                values.append(previous + 1 if comparison_true else previous - 1)
            else:
                values.append(previous - 1 if comparison_true else previous + 1)
        return tuple(values[:iteration_count])

    def _projection_value(
        self,
        item: CorrelatedCollectionItem,
        projection: tuple[str, ...],
    ) -> int | float | str | bool:
        if projection == ("$key",):
            return item.collection_key
        if projection[-1] == self.nested_key_attribute:
            return item.collection_key
        if projection[-1] == self.nested_value_attribute:
            return dict(item.projections).get(
                self.selected_value_projection or (self.nested_value_attribute,),
                _canonical_setup_value(
                    {entry.name: entry.type_name for entry in self.nested_constructor}[
                        self.nested_value_attribute
                    ]
                ),
            )
        if projection[-1] == self.setup_quantity_parameter or len(projection) == 1:
            return 1
        raise SafeSetupPlanRejection(
            SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
            "Record projection correlated witness'a bağlanamadı.",
        )

    def _setup_plan(self, witness: CorrelatedCollectionWitness) -> SafeObjectSetupPlan:
        root_arguments: list[tuple[str, int | float | str | bool]] = []
        for parameter in self.method_spec.constructor_parameters:
            root_arguments.append((parameter.name, _canonical_setup_value(parameter.type_name)))
        root = SafeObjectSlot(
            slot_id="root",
            blueprint=SafeObjectConstructionBlueprint(
                module_identity=self.module_identity,
                class_name=self.method_spec.class_name,
                constructor_arguments=tuple(root_arguments),
                class_fingerprint=self.root_class_fingerprint,
            ),
            proven_empty_attributes=(self.collection_attribute,),
        )
        slots: list[SafeObjectSlot] = [root]
        calls: list[SafeSetupCall] = []
        nested_types = {item.name: item.type_name for item in self.nested_constructor}
        for item in witness.items:
            projected_value = dict(item.projections).get(
                self.selected_value_projection or (self.nested_value_attribute,),
                _canonical_setup_value(nested_types[self.nested_value_attribute]),
            )
            arguments: list[tuple[str, int | float | str | bool]] = []
            for constructor in self.nested_constructor:
                if constructor.name == self.nested_key_attribute:
                    value = item.collection_key
                elif constructor.name == self.nested_value_attribute:
                    value = _coerce_setup_value(projected_value, constructor.type_name)
                elif constructor.name == self.nested_stock_attribute:
                    value = _coerce_setup_value(1, constructor.type_name)
                else:
                    value = _canonical_setup_value(constructor.type_name)
                arguments.append((constructor.name, value))
            slot_id = f"item_{item.item_index}"
            slots.append(
                SafeObjectSlot(
                    slot_id=slot_id,
                    blueprint=SafeObjectConstructionBlueprint(
                        module_identity=self.module_identity,
                        class_name=self.nested_class_name,
                        constructor_arguments=tuple(arguments),
                        class_fingerprint=self.nested_class_fingerprint,
                    ),
                    relation_depth=1,
                    owner_slot_id="root",
                )
            )
            calls.append(
                SafeSetupCall(
                    receiver_slot_id="root",
                    method_summary=self.setup_summary,
                    arguments=(
                        SafeTypedArgumentBinding(
                            parameter_name=self.setup_object_parameter,
                            type_name="object",
                            object_slot_id=slot_id,
                        ),
                        SafeTypedArgumentBinding(
                            parameter_name=self.setup_quantity_parameter,
                            type_name=self.setup_quantity_type,
                            value=(1 if self.setup_quantity_type == "int" else 1.0),
                        ),
                    ),
                    selected_guard_outcomes=tuple(
                        (index, False) for index in range(len(self.setup_summary.guards))
                    ),
                )
            )
        budgets = replace(
            DEFAULT_SAFE_METHOD_BUDGETS,
            guards=max(DEFAULT_SAFE_METHOD_BUDGETS.guards, len(calls) * len(self.setup_summary.guards)),
            state_effects=max(
                DEFAULT_SAFE_METHOD_BUDGETS.state_effects,
                len(calls) * self.setup_summary.budget_footprint.state_effects,
            ),
            observer_calls=max(
                DEFAULT_SAFE_METHOD_BUDGETS.observer_calls,
                len(calls) * self.setup_summary.budget_footprint.observer_calls,
            ),
            executable_statements=max(
                DEFAULT_SAFE_METHOD_BUDGETS.executable_statements,
                len(calls) * self.setup_summary.budget_footprint.executable_statements,
            ),
        )
        return SafeObjectSetupPlan(
            module_identity=self.module_identity,
            target_identity=self.target_identity,
            object_slots=tuple(slots),
            setup_calls=tuple(calls),
            receiver_slot_id="root",
            budgets=budgets,
        )


@dataclass(frozen=True, slots=True)
class SafeObjectSetupContext:
    """Path marker'larını bounded setup proof'una bağlayan internal context."""

    module_identity: str = field(repr=False)
    target_identity: str
    root_parameter_name: str
    collection_attribute: str
    root_spec: SafeObjectParameterSpec = field(repr=False)
    nested_class_name: str
    nested_constructor: tuple[_ConstructorArgumentProof, ...] = field(repr=False)
    nested_class_fingerprint: str = field(repr=False)
    nested_key_attribute: str
    nested_value_attribute: str
    nested_stock_attribute: str
    setup_object_parameter: str
    setup_quantity_parameter: str
    setup_quantity_type: str
    setup_summary: SafeMethodSummary = field(repr=False)
    observers: tuple[_ObserverProof, ...] = field(repr=False)
    state_analysis_name: str
    semantic_shape_digest: str = field(repr=False)

    @property
    def method_analysis_names(self) -> dict[str, str]:
        return {
            observer.method_name: observer.analysis_name
            for observer in self.observers
        }

    def analysis_metadata(
        self,
        parameter_names: tuple[str, ...],
        parameter_types: dict[str, str],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        names = list(parameter_names)
        types = dict(parameter_types)
        for observer in self.observers:
            for name, type_name in observer.target_argument_types:
                if name in names:
                    types[name] = type_name
        for name, type_name in (
            (self.state_analysis_name, "bool"),
            *((item.analysis_name, item.return_type) for item in self.observers),
        ):
            if name not in names:
                names.append(name)
            types[name] = type_name
        return tuple(names), types

    def rewrite_path(self, path: ExecutionPath) -> ExecutionPath:
        if not isinstance(path, ExecutionPath):
            raise TypeError("path ExecutionPath olmalıdır.")
        if "Exception" in path.edge_labels:
            return path
        transformer = _SetupMarkerTransformer(self)
        labels: list[str] = []
        for label in path.node_labels:
            if UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER not in label:
                labels.append(label)
                continue
            try:
                parsed = ast.parse(label)
            except SyntaxError:
                labels.append(label)
                continue
            rewritten = transformer.visit(parsed)
            ast.fix_missing_locations(rewritten)
            labels.append(ast.unparse(rewritten).strip())
        return ExecutionPath(
            node_ids=list(path.node_ids),
            edge_labels=list(path.edge_labels),
            node_labels=labels,
            node_types=list(path.node_types),
            line_numbers=list(path.line_numbers),
        )

    def bind_scenario(self, scenario: Any) -> Any:
        from generator.scenario_generator import Scenario

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario Scenario olmalıdır.")
        values = dict(scenario.keyword_arguments)
        state_value = values.get(self.state_analysis_name)
        if type(state_value) is not bool:
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Path-scoped collection state exact bool olmalıdır.",
            )

        root_arguments: list[tuple[str, int | float | str | bool]] = []
        for constructor in self.root_spec.constructor_parameters:
            value = values.get(constructor.analysis_name)
            expected = _PRIMITIVE_TYPES.get(constructor.type_name)
            if expected is None or type(value) is not expected:
                _reject(
                    SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
                    "Root constructor primitive binding çözümlenemedi.",
                )
            root_arguments.append((constructor.name, value))
        root_blueprint = SafeObjectConstructionBlueprint(
            module_identity=self.module_identity,
            class_name=self.root_spec.class_name,
            constructor_arguments=tuple(root_arguments),
            depth=self.root_spec.depth,
            class_fingerprint=self.setup_summary.receiver.class_fingerprint,
        )
        slots: list[SafeObjectSlot] = [
            SafeObjectSlot(
                slot_id="root",
                blueprint=root_blueprint,
                proven_empty_attributes=(self.collection_attribute,),
            )
        ]
        calls: tuple[SafeSetupCall, ...] = ()

        if state_value:
            observer_values = {
                observer.method_name: values.get(observer.analysis_name)
                for observer in self.observers
            }
            total_observer = next(
                (
                    observer
                    for observer in self.observers
                    if observer.value_attribute == self.nested_value_attribute
                ),
                None,
            )
            key_observer = next(
                (
                    observer
                    for observer in self.observers
                    if observer.value_attribute == self.nested_key_attribute
                ),
                None,
            )
            desired_total = (
                observer_values.get(total_observer.method_name)
                if total_observer is not None
                else None
            )
            desired_key = (
                observer_values.get(key_observer.method_name)
                if key_observer is not None
                else None
            )
            quantity_value: int | float = (
                1 if self.setup_quantity_type == "int" else 1.0
            )
            nested_arguments: list[tuple[str, int | float | str | bool]] = []
            for argument in self.nested_constructor:
                if argument.name == self.nested_key_attribute:
                    value = _coerce_setup_value(desired_key, argument.type_name)
                elif argument.name == self.nested_value_attribute:
                    value = _coerce_setup_value(desired_total, argument.type_name)
                elif argument.name == self.nested_stock_attribute:
                    value = _coerce_setup_value(quantity_value, argument.type_name)
                else:
                    value = _canonical_setup_value(argument.type_name)
                nested_arguments.append((argument.name, value))
            nested_blueprint = SafeObjectConstructionBlueprint(
                module_identity=self.module_identity,
                class_name=self.nested_class_name,
                constructor_arguments=tuple(nested_arguments),
                class_fingerprint=(
                    self.setup_summary.local_calls[0]
                    .callee_summary.receiver.class_fingerprint
                ),
            )
            slots.append(
                SafeObjectSlot(
                    slot_id="nested",
                    blueprint=nested_blueprint,
                    relation_depth=1,
                    owner_slot_id="root",
                )
            )
            calls = (
                SafeSetupCall(
                    receiver_slot_id="root",
                    method_summary=self.setup_summary,
                    arguments=(
                        SafeTypedArgumentBinding(
                            parameter_name=self.setup_object_parameter,
                            type_name="object",
                            object_slot_id="nested",
                        ),
                        SafeTypedArgumentBinding(
                            parameter_name=self.setup_quantity_parameter,
                            type_name=self.setup_quantity_type,
                            value=quantity_value,
                        ),
                    ),
                    selected_guard_outcomes=tuple(
                        (index, False)
                        for index in range(len(self.setup_summary.guards))
                    ),
                ),
            )

        plan = SafeObjectSetupPlan(
            module_identity=self.module_identity,
            target_identity=self.target_identity,
            object_slots=tuple(slots),
            setup_calls=calls,
            target_bindings=(
                SafeTargetBinding(
                    parameter_name=self.root_parameter_name,
                    object_slot_id="root",
                ),
            ),
        )
        private_names = {
            self.state_analysis_name,
            *(observer.analysis_name for observer in self.observers),
            *(
                parameter.analysis_name
                for parameter in self.root_spec.constructor_parameters
            ),
        }
        neutral_types = {
            name: type_name
            for observer in self.observers
            for name, type_name in observer.target_argument_types
        }
        public_arguments: list[tuple[str, Any]] = []
        for name, value in scenario.keyword_arguments:
            if name in private_names:
                continue
            if name in neutral_types:
                value = _canonical_setup_value(neutral_types[name])
            public_arguments.append((name, value))
        return replace(
            scenario,
            keyword_arguments=tuple(public_arguments),
            setup_plan=plan,
        )

    def bind_generated_input(
        self,
        generated_input: Any,
        *,
        path: ExecutionPath | None = None,
    ) -> Any:
        """PathInputGenerator çıktısını planla atomik biçimde bağlar."""
        del path
        from generator.path_input_generator import GeneratedTestInput
        from generator.scenario_generator import Scenario

        if not isinstance(generated_input, GeneratedTestInput):
            raise TypeError("generated_input GeneratedTestInput olmalıdır.")
        internal = Scenario(
            scenario_id="internal_setup_binding",
            name="internal setup binding",
            path_index=1,
            priority_rank=1,
            priority_level="Low",
            dqm_score=0.0,
            node_ids=(1,),
            edge_labels=(),
            contains_loop=False,
            contains_exception=False,
            description="internal setup binding",
            keyword_arguments=generated_input.keyword_arguments,
            expected_result=generated_input.expected_result,
            expected_exception=generated_input.expected_exception,
        )
        bound = self.bind_scenario(internal)
        return replace(
            generated_input,
            keyword_arguments=bound.keyword_arguments,
            setup_plan=bound.setup_plan,
        )


class _SetupMarkerTransformer(ast.NodeTransformer):
    def __init__(self, context: SafeObjectSetupContext) -> None:
        self._context = context

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER
            and len(node.args) >= 3
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.args[:3]
            )
        ):
            return self.generic_visit(node)
        kind, receiver, identity = (item.value for item in node.args[:3])
        if receiver != self._context.root_parameter_name:
            return self.generic_visit(node)
        if kind == "STATE_TRUTHINESS" and identity == self._context.collection_attribute:
            return ast.copy_location(
                ast.Name(id=self._context.state_analysis_name, ctx=ast.Load()),
                node,
            )
        if kind == "METHOD":
            analysis_name = self._context.method_analysis_names.get(identity)
            if analysis_name is not None:
                return ast.copy_location(
                    ast.Name(id=analysis_name, ctx=ast.Load()),
                    node,
                )
        return self.generic_visit(node)


def analyze_safe_object_setup_context(
    tree: ast.Module,
    *,
    module_identity: str,
    target_name: str,
    custom_object_spec: SafeCustomObjectTargetSpec,
) -> tuple[SafeObjectSetupContext | None, str | None]:
    """Dar bounded object-setup dilimini yalnız statik AST proof ile kurar."""
    if not isinstance(tree, ast.Module):
        raise TypeError("tree ast.Module olmalıdır.")
    _validate_module_identity(module_identity)
    if type(target_name) is not str or not target_name:
        raise ValueError("target_name boş olmayan string olmalıdır.")
    if not isinstance(custom_object_spec, SafeCustomObjectTargetSpec):
        raise TypeError("custom_object_spec türü geçersizdir.")
    if len(custom_object_spec.object_parameters) != 1:
        return None, "UNSAFE_OBJECT_GRAPH"
    root_spec = custom_object_spec.object_parameters[0]
    empty_dict_states = tuple(
        item
        for item in root_spec.state_initializers
        if item.empty_collection_kind == "EMPTY_DICT"
    )
    if len(empty_dict_states) != 1:
        return None, "UNPROVEN_EMPTY_COLLECTION"
    collection_attribute = empty_dict_states[0].attribute_name
    classes = {
        item.name: item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
    }
    root_class = classes.get(root_spec.class_name)
    if root_class is None or root_class.bases or root_class.keywords:
        return None, "UNSAFE_ROOT_CLASS"
    target = custom_object_spec.target_node
    direct_calls = _direct_object_calls(target, root_spec.parameter_name)
    if not direct_calls or len(direct_calls) > DEFAULT_SAFE_METHOD_BUDGETS.observer_calls:
        return None, "UNPROVEN_OBSERVER_CALL"
    if len({call.func.attr for call in direct_calls}) != len(direct_calls):
        return None, "UNPROVEN_OBSERVER_CALL"

    setup_proof = _find_bounded_insert_setup(
        tree,
        module_identity=module_identity,
        classes=classes,
        root_class=root_class,
        collection_attribute=collection_attribute,
    )
    if setup_proof is None:
        return None, "UNPROVEN_BOUNDED_SETUP"
    (
        nested_class,
        nested_constructor,
        nested_key_attribute,
        nested_stock_attribute,
        setup_object_parameter,
        setup_quantity_parameter,
        setup_quantity_type,
        setup_summary,
    ) = setup_proof

    observers: list[_ObserverProof] = []
    value_attribute: str | None = None
    key_type_by_attribute = {
        item.name: item.type_name for item in nested_constructor
    }
    target_argument_occurrences: dict[str, int] = {}
    for call in direct_calls:
        method_name = call.func.attr
        method = _class_method(root_class, method_name)
        if method is None:
            return None, "UNKNOWN_METHOD"
        numeric_attribute = _bounded_numeric_observer_attribute(
            method,
            collection_attribute=collection_attribute,
        )
        returned_attribute = _bounded_single_item_observer_attribute(
            method,
            collection_attribute=collection_attribute,
        )
        if numeric_attribute is not None:
            observer_type = "float"
            value_attribute = numeric_attribute
            observed_attribute = numeric_attribute
        elif (
            returned_attribute is not None
            and key_type_by_attribute.get(returned_attribute) in _PRIMITIVE_TYPES
        ):
            observer_type = key_type_by_attribute[returned_attribute]
            observed_attribute = returned_attribute
        else:
            return None, "UNPROVEN_OBSERVER_CALL"
        method_types = _method_parameter_types(tree, method, root_class.name)
        method_parameters = tuple(argument.arg for argument in method.args.args[1:])
        if len(call.args) != len(method_parameters) or call.keywords:
            return None, "UNPROVEN_OBSERVER_BINDING"
        target_argument_types: list[tuple[str, str]] = []
        for argument, parameter_name in zip(call.args, method_parameters):
            if not isinstance(argument, ast.Name):
                return None, "UNPROVEN_OBSERVER_BINDING"
            type_name = method_types.get(parameter_name)
            if type_name not in _PRIMITIVE_TYPES:
                return None, "UNKNOWN_PRIMITIVE_DOMAIN"
            target_argument_types.append((argument.id, type_name))
            target_argument_occurrences[argument.id] = (
                target_argument_occurrences.get(argument.id, 0) + 1
            )
        observers.append(
            _ObserverProof(
                method_name=method_name,
                analysis_name="",
                return_type=observer_type,
                value_attribute=observed_attribute,
                target_argument_types=tuple(target_argument_types),
            )
        )
    if value_attribute is None:
        return None, "UNPROVEN_NUMERIC_OBSERVER"
    if value_attribute not in key_type_by_attribute:
        return None, "UNKNOWN_PRIMITIVE_DOMAIN"
    if not _target_arguments_are_observer_scoped(
        target,
        target_argument_occurrences,
    ):
        return None, "UNSAFE_TARGET_ARGUMENT_SCOPE"

    occupied_names = {
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
    }
    occupied_names.update(
        node.id for node in ast.walk(target) if isinstance(node, ast.Name)
    )
    state_name = _fresh_analysis_name("__safe_setup_state", occupied_names)
    occupied_names.add(state_name)
    bound_observers: list[_ObserverProof] = []
    for index, observer in enumerate(observers):
        analysis_name = _fresh_analysis_name(
            f"__safe_setup_observer_{index}",
            occupied_names,
        )
        occupied_names.add(analysis_name)
        bound_observers.append(replace(observer, analysis_name=analysis_name))
    semantic_shape = _fingerprint(
        (
            "BOUNDED_OBJECT_SETUP",
            tuple(item.type_name for item in nested_constructor),
            setup_quantity_type,
            tuple(item.return_type for item in bound_observers),
            len(bound_observers),
        )
    )
    return (
        SafeObjectSetupContext(
            module_identity=module_identity,
            target_identity=target_name,
            root_parameter_name=root_spec.parameter_name,
            collection_attribute=collection_attribute,
            root_spec=root_spec,
            nested_class_name=nested_class.name,
            nested_constructor=nested_constructor,
            nested_class_fingerprint=_fingerprint(
                (
                    "SAFE_NESTED_CLASS",
                    nested_class.name,
                    tuple(
                        (argument.name, argument.type_name)
                        for argument in nested_constructor
                    ),
                )
            ),
            nested_key_attribute=nested_key_attribute,
            nested_value_attribute=value_attribute,
            nested_stock_attribute=nested_stock_attribute,
            setup_object_parameter=setup_object_parameter,
            setup_quantity_parameter=setup_quantity_parameter,
            setup_quantity_type=setup_quantity_type,
            setup_summary=setup_summary,
            observers=tuple(bound_observers),
            state_analysis_name=state_name,
            semantic_shape_digest=semantic_shape,
        ),
        None,
    )


def analyze_safe_method_collection_context(
    tree: ast.Module,
    *,
    module_identity: str,
    target_name: str,
    method_spec: SimpleInstanceMethodSpec,
) -> tuple[SafeMethodCollectionSetupContext | None, str | None]:
    """Doğrudan method receiver'ı için üç bounded collection kalıbını kanıtlar."""
    if not isinstance(tree, ast.Module):
        raise TypeError("tree ast.Module olmalıdır.")
    _validate_module_identity(module_identity)
    if not isinstance(method_spec, SimpleInstanceMethodSpec):
        raise TypeError("method_spec SimpleInstanceMethodSpec olmalıdır.")
    if target_name != method_spec.qualified_name:
        raise ValueError("target_name method spec qualified identity ile eşleşmelidir.")
    if method_spec.constructor_parameters:
        return None, "UNPROVEN_RECEIVER_CONSTRUCTOR_STATE"
    empty_dict_states = tuple(
        name
        for name, initializer in method_spec.state_initializers
        if isinstance(initializer, ast.Dict)
        and not initializer.keys
        and not initializer.values
    )
    if len(empty_dict_states) != 1:
        return None, "UNPROVEN_EMPTY_COLLECTION"
    collection_attribute = empty_dict_states[0]
    classes = {
        item.name: item for item in tree.body if isinstance(item, ast.ClassDef)
    }
    root_class = classes.get(method_spec.class_name)
    if root_class is None or root_class.bases or root_class.keywords:
        return None, "UNSAFE_ROOT_CLASS"
    setup_proof = _find_bounded_insert_setup(
        tree,
        module_identity=module_identity,
        classes=classes,
        root_class=root_class,
        collection_attribute=collection_attribute,
    )
    if setup_proof is None:
        return None, "UNPROVEN_BOUNDED_SETUP"
    (
        nested_class,
        nested_constructor,
        nested_key_attribute,
        nested_stock_attribute,
        setup_object_parameter,
        setup_quantity_parameter,
        setup_quantity_type,
        setup_summary,
    ) = setup_proof
    pattern = _analyze_collection_method_pattern(
        method_spec.method_node,
        collection_attribute=collection_attribute,
        nested_constructor=nested_constructor,
        nested_key_attribute=nested_key_attribute,
        nested_stock_attribute=nested_stock_attribute,
        setup_quantity_parameter=setup_quantity_parameter,
    )
    if pattern is None:
        return None, "UNPROVEN_CORRELATED_COLLECTION_PATTERN"
    occupied = {
        node.id
        for node in ast.walk(method_spec.method_node)
        if isinstance(node, ast.Name)
    }
    aggregate_name = None
    iterable_name = None
    if pattern["pattern_kind"] == "SCALAR_FOLD":
        aggregate_name = _fresh_analysis_name(
            "__safe_collection_aggregate", occupied
        )
        occupied.add(aggregate_name)
    iterable_name = _fresh_analysis_name(
        "__safe_collection_values", occupied
    )
    occupied.add(iterable_name)
    initial_condition_names: tuple[str, ...] = ()
    selection_condition_names: tuple[str, ...] = ()
    if pattern["pattern_kind"] == "ARG_EXTREME":
        generated_initial_names: list[str] = []
        for index in range(2):
            name = _fresh_analysis_name(
                f"__safe_selection_initial_{index}", occupied
            )
            occupied.add(name)
            generated_initial_names.append(name)
        initial_condition_names = tuple(generated_initial_names)
        generated_selection_names: list[str] = []
        for index in range(2):
            name = _fresh_analysis_name(
                f"__safe_selection_compare_{index}", occupied
            )
            occupied.add(name)
            generated_selection_names.append(name)
        selection_condition_names = tuple(generated_selection_names)
    return (
        SafeMethodCollectionSetupContext(
            module_identity=module_identity,
            target_identity=target_name,
            method_spec=method_spec,
            collection_attribute=collection_attribute,
            root_class_fingerprint=setup_summary.receiver.class_fingerprint,
            nested_class_name=nested_class.name,
            nested_constructor=nested_constructor,
            nested_class_fingerprint=(
                setup_summary.local_calls[0].callee_summary.receiver.class_fingerprint
            ),
            nested_key_attribute=nested_key_attribute,
            nested_value_attribute=pattern["nested_value_attribute"],
            nested_stock_attribute=nested_stock_attribute,
            setup_object_parameter=setup_object_parameter,
            setup_quantity_parameter=setup_quantity_parameter,
            setup_quantity_type=setup_quantity_type,
            setup_summary=setup_summary,
            item_projection_paths=pattern["item_projection_paths"],
            record_projection_paths=pattern["record_projection_paths"],
            selection_operator=pattern["selection_operator"],
            selected_identity_projection=pattern["selected_identity_projection"],
            selected_value_projection=pattern["selected_value_projection"],
            aggregate_analysis_name=aggregate_name,
            iterable_analysis_name=iterable_name,
            loop_target_name=(
                pattern["loop_target_name"]
                if pattern["pattern_kind"] == "ARG_EXTREME"
                else "__safe_collection_index"
            ),
            accumulator_name=pattern["accumulator_name"],
            selected_name=pattern["selected_name"],
            item_object_name=pattern["item_object_name"],
            state_assignment_label=f"__self_{collection_attribute} = {{}}",
            loop_label=pattern["loop_label"],
            neutral_assignment_labels=pattern["neutral_assignment_labels"],
            update_label=pattern["update_label"],
            state_condition_label=(
                f"not __self_{collection_attribute}"
                if pattern["state_condition_label"] is not None
                else None
            ),
            initial_selection_condition_label=pattern.get(
                "initial_selection_condition_label"
            ),
            selection_condition_label=pattern["selection_condition_label"],
            initial_condition_analysis_names=initial_condition_names,
            selection_condition_analysis_names=selection_condition_names,
            return_label=pattern["return_label"],
            append_label=pattern["append_label"],
            pattern_kind=pattern["pattern_kind"],
        ),
        None,
    )


def _analyze_collection_method_pattern(
    method: ast.FunctionDef,
    *,
    collection_attribute: str,
    nested_constructor: tuple[_ConstructorArgumentProof, ...],
    nested_key_attribute: str,
    nested_stock_attribute: str,
    setup_quantity_parameter: str,
) -> dict[str, Any] | None:
    loops = [node for node in method.body if isinstance(node, ast.For)]
    if len(loops) != 1:
        return None
    loop = loops[0]
    view = _safe_collection_view(loop.iter, collection_attribute)
    if view not in {"values", "items"} or loop.orelse:
        return None
    loop_label = f"{ast.unparse(loop.target)} in ()"
    aliases: dict[str, tuple[str, ...]] = {}
    projection_target = (
        loop.target.elts[1]
        if isinstance(loop.target, ast.Tuple) and len(loop.target.elts) == 2
        else loop.target
    )
    neutral_labels: list[str] = []
    for statement in loop.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            path = _safe_item_projection(
                statement.value,
                projection_target,
                aliases,
            )
            if path is not None:
                aliases[statement.targets[0].id] = path
                neutral_labels.append(ast.unparse(statement))

    constructor_names = {item.name for item in nested_constructor}
    numeric_updates = [
        node for node in loop.body
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.op, ast.Add)
    ]
    if len(numeric_updates) == 1:
        update = numeric_updates[0]
        projections = tuple(
            path
            for node in _flatten_product(update.value)
            if (path := _safe_item_projection(node, projection_target, aliases)) is not None
        )
        if len(projections) == 2:
            value_paths = tuple(
                path for path in projections if path[-1] in constructor_names
            )
            value_path = next(
                (
                    path for path in value_paths
                    if path[-1] not in {nested_key_attribute, nested_stock_attribute}
                ),
                None,
            )
            if value_path is not None and _has_zero_initializer(
                method.body, update.target.id, before=loop
            ):
                return_node = _single_return(method.body)
                if return_node is not None:
                    return {
                        "pattern_kind": "SCALAR_FOLD",
                        "nested_value_attribute": value_path[-1],
                        "item_projection_paths": projections,
                        "record_projection_paths": (),
                        "selection_operator": None,
                        "selected_identity_projection": None,
                        "selected_value_projection": value_path,
                        "loop_target_name": ast.unparse(loop.target),
                        "accumulator_name": update.target.id,
                        "selected_name": None,
                        "item_object_name": None,
                        "loop_label": loop_label,
                        "neutral_assignment_labels": tuple(neutral_labels),
                        "update_label": ast.unparse(update),
                        "state_condition_label": None,
                        "selection_condition_label": None,
                        "return_label": ast.unparse(return_node),
                        "append_label": None,
                    }

    if view == "values":
        selection = _arg_extreme_pattern(
            method,
            loop,
            aliases=aliases,
            collection_attribute=collection_attribute,
            constructor_names=constructor_names,
            nested_key_attribute=nested_key_attribute,
        )
        if selection is not None:
            selection.update(
                loop_label=loop_label,
                neutral_assignment_labels=tuple(neutral_labels),
                item_projection_paths=(
                    selection["selected_identity_projection"],
                    selection["selected_value_projection"],
                ),
                record_projection_paths=(),
                accumulator_name=None,
                update_label=None,
                append_label=None,
            )
            return selection

    if view == "items" and isinstance(loop.target, ast.Tuple):
        projection = _record_projection_pattern(
            method,
            loop,
            aliases=aliases,
            constructor_names=constructor_names,
            nested_key_attribute=nested_key_attribute,
            setup_quantity_parameter=setup_quantity_parameter,
        )
        if projection is not None:
            projection.update(
                loop_label=loop_label,
                neutral_assignment_labels=tuple(neutral_labels),
                item_projection_paths=tuple(
                    path for _, path in projection["record_projection_paths"]
                ),
                selection_operator=None,
                selected_identity_projection=None,
                selected_value_projection=next(
                    (
                        path
                        for _, path in projection["record_projection_paths"]
                        if path[-1] in constructor_names
                        and path[-1] not in {nested_key_attribute, nested_stock_attribute}
                    ),
                    None,
                ),
                accumulator_name=None,
                selected_name=None,
                item_object_name=None,
                update_label=None,
                state_condition_label=None,
                selection_condition_label=None,
            )
            if projection["selected_value_projection"] is not None:
                projection["nested_value_attribute"] = projection[
                    "selected_value_projection"
                ][-1]
                return projection
    return None


def _arg_extreme_pattern(
    method: ast.FunctionDef,
    loop: ast.For,
    *,
    aliases: dict[str, tuple[str, ...]],
    collection_attribute: str,
    constructor_names: set[str],
    nested_key_attribute: str,
) -> dict[str, Any] | None:
    selected_init = next(
        (
            node for node in method.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ),
        None,
    )
    if selected_init is None:
        return None
    selected_name = selected_init.targets[0].id
    initial_checks = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is)
        and isinstance(node.left, ast.Name)
        and node.left.id == selected_name
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    ]
    if len(initial_checks) != 1:
        return None
    object_alias = next(
        (
            name for name, path in aliases.items()
            if len(path) == 1
        ),
        None,
    )
    if object_alias is None:
        return None
    comparisons = [
        node for node in ast.walk(loop)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Gt, ast.Lt))
    ]
    if len(comparisons) != 1:
        return None
    comparison = comparisons[0]
    left_path = _safe_item_projection(comparison.left, loop.target, aliases)
    right = comparison.comparators[0]
    if not (
        left_path is not None
        and left_path[-1] in constructor_names
        and isinstance(right, ast.Attribute)
        and isinstance(right.value, ast.Name)
        and right.value.id == selected_name
        and right.attr == left_path[-1]
    ):
        return None
    return_node = _single_return(method.body)
    if not (
        return_node is not None
        and isinstance(return_node.value, ast.Attribute)
        and isinstance(return_node.value.value, ast.Name)
        and return_node.value.value.id == selected_name
        and return_node.value.attr == nested_key_attribute
    ):
        return None
    state_if = next(
        (
            node for node in method.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and _is_self_collection_attribute(node.test.operand, collection_attribute)
        ),
        None,
    )
    if state_if is None:
        return None
    operator = ">" if isinstance(comparison.ops[0], ast.Gt) else "<"
    return {
        "pattern_kind": "ARG_EXTREME",
        "nested_value_attribute": left_path[-1],
        "selection_operator": operator,
        "selected_identity_projection": (*aliases[object_alias], nested_key_attribute),
        "selected_value_projection": left_path,
        "loop_target_name": object_alias,
        "selected_name": selected_name,
        "item_object_name": object_alias,
        "state_condition_label": ast.unparse(state_if.test),
        "initial_selection_condition_label": ast.unparse(initial_checks[0]),
        "selection_condition_label": ast.unparse(comparison),
        "return_label": ast.unparse(return_node),
    }


def _record_projection_pattern(
    method: ast.FunctionDef,
    loop: ast.For,
    *,
    aliases: dict[str, tuple[str, ...]],
    constructor_names: set[str],
    nested_key_attribute: str,
    setup_quantity_parameter: str,
) -> dict[str, Any] | None:
    if len(loop.target.elts) != 2 or not all(
        isinstance(item, ast.Name) for item in loop.target.elts
    ):
        return None
    key_name = loop.target.elts[0].id
    record_name = loop.target.elts[1].id
    append = next(
        (
            node for node in loop.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "append"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Dict)
        ),
        None,
    )
    if append is None:
        return None
    result_name = append.value.func.value
    if not isinstance(result_name, ast.Name):
        return None
    return_node = _single_return(method.body)
    if not (
        return_node is not None
        and isinstance(return_node.value, ast.Name)
        and return_node.value.id == result_name.id
    ):
        return None
    projections: list[tuple[str, tuple[str, ...]]] = []
    payload = append.value.args[0]
    for key, value in zip(payload.keys, payload.values):
        if not isinstance(key, ast.Constant) or type(key.value) is not str:
            return None
        if isinstance(value, ast.Name) and value.id == key_name:
            path = ("$key",)
        else:
            path = _safe_item_projection(
                value,
                ast.Name(id=record_name, ctx=ast.Load()),
                aliases,
            )
        if path is None:
            return None
        projections.append((key.value, path))
    if not any(path[-1] in constructor_names for _, path in projections):
        return None
    return {
        "pattern_kind": "RECORD_PROJECTION",
        "record_projection_paths": tuple(projections),
        "loop_target_name": ast.unparse(loop.target),
        "return_label": ast.unparse(return_node),
        "append_label": ast.unparse(append),
    }


def _safe_collection_view(node: ast.expr, attribute: str) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and _is_self_collection_attribute(node.func.value, attribute)
    ):
        return None
    return node.func.attr


def _is_self_collection_attribute(node: ast.AST, attribute: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attribute
    )


def _safe_item_projection(
    node: ast.expr,
    loop_target: ast.expr,
    aliases: dict[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        if isinstance(loop_target, ast.Name) and node.id == loop_target.id:
            return ()
        return aliases.get(node.id)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is str
    ):
        base = _safe_item_projection(node.value, loop_target, aliases)
        return (*base, node.slice.value) if base is not None else None
    if isinstance(node, ast.Attribute):
        base = _safe_item_projection(node.value, loop_target, aliases)
        return (*base, node.attr) if base is not None else None
    return None


def _flatten_product(node: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return (*_flatten_product(node.left), *_flatten_product(node.right))
    return (node,)


def _has_zero_initializer(
    statements: list[ast.stmt],
    name: str,
    *,
    before: ast.stmt,
) -> bool:
    for statement in statements:
        if statement is before:
            break
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) in {int, float}
            and statement.value.value == 0
        ):
            return True
    return False


def _single_return(statements: list[ast.stmt]) -> ast.Return | None:
    values = [node for node in statements if isinstance(node, ast.Return)]
    return values[-1] if values else None


def _path_iteration_count(path: ExecutionPath, loop_label: str) -> int:
    del loop_label
    return sum(edge == "Iterate" for edge in path.edge_labels)


def _indexed_setup_value(type_name: str, index: int) -> int | float | str | bool:
    if type_name == "int":
        return index + 1
    if type_name == "float":
        return float(index + 1)
    if type_name == "str":
        return f"item_{index + 1}"
    if type_name == "bool":
        return bool(index)
    _reject(
        SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
        "Collection key primitive domain'i çözümlenemedi.",
    )
    raise AssertionError("unreachable")


def _find_bounded_insert_setup(
    tree: ast.Module,
    *,
    module_identity: str,
    classes: dict[str, ast.ClassDef],
    root_class: ast.ClassDef,
    collection_attribute: str,
) -> tuple[
    ast.ClassDef,
    tuple[_ConstructorArgumentProof, ...],
    str,
    str,
    str,
    str,
    str,
    SafeMethodSummary,
] | None:
    bundle = analyze_safe_method_summaries(
        tree,
        module_identity=module_identity,
    )
    candidates: list[tuple[Any, ...]] = []
    for method in root_class.body:
        if not isinstance(method, ast.FunctionDef) or method.name == "__init__":
            continue
        if (
            not _has_exact_instance_signature(method, parameter_count=2)
            or method.decorator_list
            or _contains_unsafe_setup_nodes(method)
        ):
            continue
        insertion = _single_empty_dict_insertion(method, collection_attribute)
        if insertion is None:
            continue
        object_parameter, quantity_parameter, key_attribute = insertion
        nested_call = _single_nested_mutator_call(
            method,
            object_parameter=object_parameter,
            quantity_parameter=quantity_parameter,
        )
        if nested_call is None:
            continue
        nested_method_name = nested_call
        stock_attribute = _guarded_numeric_attribute(
            method,
            object_parameter=object_parameter,
            quantity_parameter=quantity_parameter,
        )
        if stock_attribute is None:
            continue
        root_guards = _bounded_root_guards(
            method,
            collection_attribute=collection_attribute,
            object_parameter=object_parameter,
            quantity_parameter=quantity_parameter,
            key_attribute=key_attribute,
            stock_attribute=stock_attribute,
            nested_method_name=nested_method_name,
        )
        if root_guards is None:
            continue
        if not _root_setup_calls_are_safe(
            method,
            object_parameter=object_parameter,
            quantity_parameter=quantity_parameter,
            nested_method_name=nested_method_name,
        ):
            continue
        if not _root_collection_writes_are_bounded(
            method,
            collection_attribute=collection_attribute,
            object_parameter=object_parameter,
            quantity_parameter=quantity_parameter,
            key_attribute=key_attribute,
        ):
            continue
        quantity_types = _method_parameter_types(tree, method, root_class.name)
        quantity_type = quantity_types.get(quantity_parameter)
        if quantity_type not in {"int", "float"}:
            continue
        nested_classes = tuple(
            candidate
            for candidate in classes.values()
            if candidate is not root_class
            and _is_safe_nested_constructor(candidate)
            and _class_method(candidate, nested_method_name) is not None
            and _constructor_assigns_attributes(
                candidate,
                frozenset({key_attribute, stock_attribute}),
            )
        )
        if len(nested_classes) != 1:
            continue
        nested_class = nested_classes[0]
        nested_method = _class_method(nested_class, nested_method_name)
        assert nested_method is not None
        if not _is_safe_numeric_mutator(
            nested_method,
            stock_attribute=stock_attribute,
        ):
            continue
        nested_guards = _bounded_nested_guards(
            nested_method,
            stock_attribute=stock_attribute,
        )
        if nested_guards is None:
            continue
        constructor = _constructor_argument_proofs(tree, nested_class)
        if constructor is None:
            continue
        try:
            original_root = bundle.summary_for(f"{root_class.name}.{method.name}")
            original_nested = bundle.summary_for(
                f"{nested_class.name}.{nested_method.name}"
            )
        except ValueError:
            continue
        if not (
            original_root.receiver.class_namespace_safe
            and original_nested.receiver.class_namespace_safe
        ):
            continue
        root_receiver = replace(
            original_root.receiver,
            constructor_safe=True,
        )
        nested_receiver = replace(
            original_nested.receiver,
            constructor_safe=True,
        )
        numeric_domain = SafePrimitiveDomain(
            (quantity_type,),
            ("BOUNDED_SETUP_ARGUMENT",),
            1 if quantity_type == "int" else 1.0,
        )
        stock_constructor_index = next(
            index
            for index, item in enumerate(constructor)
            if item.name == stock_attribute
        )
        key_constructor_index = next(
            index
            for index, item in enumerate(constructor)
            if item.name == key_attribute
        )
        zero_value: int | float = 0 if quantity_type == "int" else 0.0
        root_guards = (
            SafeGuardSummary(
                kind="NUMERIC_BOUND",
                operator="<=",
                branch_polarities=(True, False),
                expression_shape=(
                    "COMPARE",
                    ("<=",),
                    ("NAME", quantity_parameter),
                    (("LITERAL", (quantity_type, zero_value)),),
                ),
            ),
            SafeGuardSummary(
                kind="NESTED_CAPACITY",
                operator="<",
                branch_polarities=(True, False),
                expression_shape=(
                    "COMPARE",
                    ("<",),
                    (
                        "OBJECT_CONSTRUCTOR_ARGUMENT",
                        0,
                        stock_constructor_index,
                    ),
                    (("NAME", quantity_parameter),),
                ),
            ),
            SafeGuardSummary(
                kind="EMPTY_DICT_MEMBERSHIP",
                operator="in",
                branch_polarities=(True, False),
                expression_shape=(
                    "COMPARE",
                    ("in",),
                    (
                        "OBJECT_CONSTRUCTOR_ARGUMENT",
                        0,
                        key_constructor_index,
                    ),
                    (("PROVEN_EMPTY_COLLECTION", 0),),
                ),
            ),
        )
        nested_summary = SafeMethodSummary(
            receiver=nested_receiver,
            signature=((nested_method.args.args[1].arg, numeric_domain),),
            category=SafeMethodCategory.MUTATOR,
            guards=nested_guards,
            state_effects=(
                SafeStateEffect(
                    kind="ATTRIBUTE_AUG_ASSIGN",
                    attribute_name=stock_attribute,
                    value_domain=numeric_domain,
                    source_kind="PARAMETER",
                    source_name=nested_method.args.args[1].arg,
                    expression_shape=("STATE_PLUS_PARAMETER",),
                ),
            ),
            local_calls=(),
            return_summary=SafeReturnSummary(kind="NONE"),
            operation_kinds=("GUARD", "ATTRIBUTE_AUG_ASSIGN"),
            budgets=DEFAULT_SAFE_METHOD_BUDGETS,
            rejection_category=None,
            statement_count=len(nested_guards) + 1,
            observer_call_count=0,
        )
        key_type = next(
            item.type_name for item in constructor if item.name == key_attribute
        )
        key_domain = SafePrimitiveDomain(
            (key_type,),
            ("BOUNDED_DICT_KEY",),
            _canonical_setup_value(key_type),
        )
        local_call = SafeLocalCallSummary(
            method_identity=nested_method.name,
            argument_domains=(numeric_domain,),
            call_depth=1,
            callee_summary=nested_summary,
            argument_shapes=(("NEGATED_PARAMETER",),),
            guard_context=((0, False), (1, False)),
        )
        setup_summary = SafeMethodSummary(
            receiver=root_receiver,
            signature=((quantity_parameter, numeric_domain),),
            category=SafeMethodCategory.COMPOSED,
            guards=root_guards,
            state_effects=(
                SafeStateEffect(
                    kind="DICT_SET",
                    attribute_name=collection_attribute,
                    source_kind="OWNED_RECORD",
                    key_domain=key_domain,
                    key_source_kind="NESTED_ATTRIBUTE",
                    key_source_name=key_attribute,
                    expression_shape=("OWNED_OBJECT", "PRIMITIVE_QUANTITY"),
                    guard_context=((0, False), (1, False), (2, False)),
                ),
            ),
            local_calls=(local_call,),
            return_summary=SafeReturnSummary(
                kind="LITERAL",
                domain=SafePrimitiveDomain(
                    ("bool",),
                    ("RETURN_LITERAL",),
                    True,
                ),
                literal_values=(("bool", True),),
                guard_context=((0, False), (1, False), (2, False)),
            ),
            operation_kinds=(
                "GUARD",
                "DICT_SET",
                "LOCAL_SAFE_CALL",
                "RETURN_LITERAL",
            ),
            budgets=DEFAULT_SAFE_METHOD_BUDGETS,
            rejection_category=None,
            statement_count=len(root_guards) + 2,
            observer_call_count=0,
        )
        candidates.append(
            (
                nested_class,
                constructor,
                key_attribute,
                stock_attribute,
                object_parameter,
                quantity_parameter,
                quantity_type,
                setup_summary,
            )
        )
    return candidates[0] if len(candidates) == 1 else None


def _direct_object_calls(
    target: ast.FunctionDef,
    parameter_name: str,
) -> tuple[ast.Call, ...]:
    return tuple(
        sorted(
            (
                node
                for node in ast.walk(target)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == parameter_name
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
    )


def _class_method(
    class_node: ast.ClassDef,
    method_name: str,
) -> ast.FunctionDef | None:
    candidates = tuple(
        item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == method_name
    )
    if len(candidates) != 1 or candidates[0].decorator_list:
        return None
    return candidates[0]


def _contains_unsafe_setup_nodes(method: ast.FunctionDef) -> bool:
    return any(
        isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Delete,
                ast.Global,
                ast.Import,
                ast.ImportFrom,
                ast.Lambda,
                ast.Nonlocal,
                ast.Try,
                ast.While,
                ast.With,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for node in ast.walk(method)
        if node is not method
    )


def _has_exact_instance_signature(
    method: ast.FunctionDef,
    *,
    parameter_count: int,
) -> bool:
    arguments = method.args
    return (
        not arguments.posonlyargs
        and not arguments.kwonlyargs
        and arguments.vararg is None
        and arguments.kwarg is None
        and len(arguments.args) == parameter_count + 1
        and arguments.args[0].arg == "self"
        and all(
            primitive_literal_type(default) is not None
            for default in arguments.defaults
        )
    )


def _single_empty_dict_insertion(
    method: ast.FunctionDef,
    collection_attribute: str,
) -> tuple[str, str, str] | None:
    parameter_names = {item.arg for item in method.args.args[1:]}
    candidates: list[tuple[str, str, str]] = []
    for node in ast.walk(method):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Attribute)
            and isinstance(node.targets[0].value.value, ast.Name)
            and node.targets[0].value.value.id == "self"
            and node.targets[0].value.attr == collection_attribute
            and isinstance(node.value, ast.Dict)
        ):
            continue
        slice_expression = node.targets[0].slice
        if not (
            isinstance(slice_expression, ast.Attribute)
            and isinstance(slice_expression.value, ast.Name)
            and slice_expression.value.id in parameter_names
        ):
            continue
        object_parameter = slice_expression.value.id
        value_names = {
            item.id for item in node.value.values if isinstance(item, ast.Name)
        }
        quantity_names = value_names & (parameter_names - {object_parameter})
        if object_parameter not in value_names or len(quantity_names) != 1:
            continue
        candidates.append(
            (object_parameter, next(iter(quantity_names)), slice_expression.attr)
        )
    return candidates[0] if len(candidates) == 1 else None


def _single_nested_mutator_call(
    method: ast.FunctionDef,
    *,
    object_parameter: str,
    quantity_parameter: str,
) -> str | None:
    candidates: list[str] = []
    for node in ast.walk(method):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == object_parameter
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.UnaryOp)
            and isinstance(node.args[0].op, ast.USub)
            and isinstance(node.args[0].operand, ast.Name)
            and node.args[0].operand.id == quantity_parameter
        ):
            continue
        candidates.append(node.func.attr)
    return candidates[0] if len(candidates) == 1 else None


def _guarded_numeric_attribute(
    method: ast.FunctionDef,
    *,
    object_parameter: str,
    quantity_parameter: str,
) -> str | None:
    candidates: set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Compare):
            continue
        attributes = tuple(
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and item.value.id == object_parameter
        )
        names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        if quantity_parameter in names:
            candidates.update(item.attr for item in attributes)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _bounded_root_guards(
    method: ast.FunctionDef,
    *,
    collection_attribute: str,
    object_parameter: str,
    quantity_parameter: str,
    key_attribute: str,
    stock_attribute: str,
    nested_method_name: str,
) -> tuple[SafeGuardSummary, ...] | None:
    if len(method.body) != 5:
        return None
    quantity_guard, capacity_guard, membership_guard, nested_call, returned = (
        method.body
    )
    if not (
        isinstance(quantity_guard, ast.If)
        and _is_single_compare(
            quantity_guard.test,
            left=lambda node: isinstance(node, ast.Name)
            and node.id == quantity_parameter,
            operator=ast.LtE,
            right=_is_numeric_zero,
        )
        and _is_raise_only_guard(quantity_guard)
        and isinstance(capacity_guard, ast.If)
        and _is_single_compare(
            capacity_guard.test,
            left=lambda node: isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == object_parameter
            and node.attr == stock_attribute,
            operator=ast.Lt,
            right=lambda node: isinstance(node, ast.Name)
            and node.id == quantity_parameter,
        )
        and len(capacity_guard.body) == 1
        and isinstance(capacity_guard.body[0], ast.Return)
        and isinstance(capacity_guard.body[0].value, ast.Constant)
        and capacity_guard.body[0].value.value is False
        and not capacity_guard.orelse
        and isinstance(membership_guard, ast.If)
        and _is_single_compare(
            membership_guard.test,
            left=lambda node: isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == object_parameter
            and node.attr == key_attribute,
            operator=ast.In,
            right=lambda node: isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == collection_attribute,
        )
        and len(membership_guard.body) == 1
        and isinstance(membership_guard.body[0], ast.AugAssign)
        and len(membership_guard.orelse) == 1
        and isinstance(membership_guard.orelse[0], ast.Assign)
        and isinstance(nested_call, ast.Expr)
        and isinstance(nested_call.value, ast.Call)
        and isinstance(nested_call.value.func, ast.Attribute)
        and isinstance(nested_call.value.func.value, ast.Name)
        and nested_call.value.func.value.id == object_parameter
        and nested_call.value.func.attr == nested_method_name
        and isinstance(returned, ast.Return)
        and isinstance(returned.value, ast.Constant)
        and returned.value.value is True
    ):
        return None
    return (
        SafeGuardSummary(
            kind="NUMERIC_BOUND",
            operator="<=",
            branch_polarities=(True, False),
            expression_shape=("PARAMETER_NON_POSITIVE",),
        ),
        SafeGuardSummary(
            kind="NESTED_CAPACITY",
            operator="<",
            branch_polarities=(True, False),
            expression_shape=("NESTED_ATTRIBUTE_COMPARE",),
        ),
        SafeGuardSummary(
            kind="EMPTY_DICT_MEMBERSHIP",
            operator="in",
            branch_polarities=(True, False),
            expression_shape=("PROVEN_EMPTY_DICT",),
        ),
    )


def _bounded_nested_guards(
    method: ast.FunctionDef,
    *,
    stock_attribute: str,
) -> tuple[SafeGuardSummary, ...] | None:
    parameters = method.args.args[1:]
    if len(parameters) != 1 or len(method.body) != 2:
        return None
    parameter_name = parameters[0].arg
    guard, update = method.body
    if not (
        isinstance(guard, ast.If)
        and _is_single_compare(
            guard.test,
            left=lambda node: isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Attribute)
            and isinstance(node.left.value, ast.Name)
            and node.left.value.id == "self"
            and node.left.attr == stock_attribute
            and isinstance(node.right, ast.Name)
            and node.right.id == parameter_name,
            operator=ast.Lt,
            right=_is_numeric_zero,
        )
        and _is_raise_only_guard(guard)
        and isinstance(update, ast.AugAssign)
    ):
        return None
    return (
        SafeGuardSummary(
            kind="NUMERIC_BOUND",
            operator="<",
            branch_polarities=(True, False),
            expression_shape=("STATE_PLUS_PARAMETER",),
        ),
    )


def _is_single_compare(
    expression: ast.expr,
    *,
    left: Any,
    operator: type[ast.cmpop],
    right: Any,
) -> bool:
    return (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], operator)
        and len(expression.comparators) == 1
        and left(expression.left)
        and right(expression.comparators[0])
    )


def _is_numeric_zero(expression: ast.expr) -> bool:
    return (
        isinstance(expression, ast.Constant)
        and type(expression.value) in {int, float}
        and expression.value == 0
    )


def _is_raise_only_guard(statement: ast.If) -> bool:
    if len(statement.body) != 1 or statement.orelse:
        return False
    raised = statement.body[0]
    return (
        isinstance(raised, ast.Raise)
        and isinstance(raised.exc, ast.Call)
        and isinstance(raised.exc.func, ast.Name)
        and raised.exc.func.id in {"ValueError", "TypeError"}
    )


def _constructor_assigns_attributes(
    class_node: ast.ClassDef,
    required: frozenset[str],
) -> bool:
    constructor = _class_method(class_node, "__init__")
    if constructor is None:
        return False
    assigned = {
        target.attr
        for statement in constructor.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and isinstance(statement.value, ast.Name)
    }
    return required <= assigned


def _is_safe_numeric_mutator(
    method: ast.FunctionDef,
    *,
    stock_attribute: str,
) -> bool:
    if (
        not _has_exact_instance_signature(method, parameter_count=1)
        or method.decorator_list
        or _contains_unsafe_setup_nodes(method)
    ):
        return False
    parameters = method.args.args[1:]
    if len(parameters) != 1:
        return False
    parameter_name = parameters[0].arg
    updates = tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
        and node.target.attr == stock_attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == parameter_name
    )
    if len(updates) != 1:
        return False
    for node in ast.walk(method):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if node is not updates[0].target:
                return False
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            return False
    allowed_calls = tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    )
    return all(
        isinstance(call.func, ast.Name)
        and call.func.id in {"ValueError", "TypeError"}
        and any(
            isinstance(parent, ast.Raise) and parent.exc is call
            for parent in ast.walk(method)
        )
        for call in allowed_calls
    )


def _constructor_argument_proofs(
    tree: ast.Module,
    class_node: ast.ClassDef,
) -> tuple[_ConstructorArgumentProof, ...] | None:
    constructor = _class_method(class_node, "__init__")
    if constructor is None or not _is_safe_nested_constructor(class_node):
        return None
    arguments = constructor.args.args[1:]
    defaults = (None,) * (len(arguments) - len(constructor.args.defaults)) + tuple(
        constructor.args.defaults
    )
    call_types = _observed_constructor_argument_types(tree, class_node.name, arguments)
    proofs: list[_ConstructorArgumentProof] = []
    for argument, default in zip(arguments, defaults):
        type_name = primitive_annotation_type(argument.annotation)
        if type_name is None and default is not None:
            type_name = primitive_literal_type(default)
        if type_name is None:
            type_name = call_types.get(argument.arg)
        if type_name not in _PRIMITIVE_TYPES:
            return None
        proofs.append(_ConstructorArgumentProof(argument.arg, type_name))
    assigned = {
        statement.targets[0].attr: statement.value.id
        for statement in constructor.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Attribute)
        and isinstance(statement.targets[0].value, ast.Name)
        and statement.targets[0].value.id == "self"
        and isinstance(statement.value, ast.Name)
    }
    if set(assigned.values()) != {item.arg for item in arguments}:
        return None
    if any(attribute != parameter for attribute, parameter in assigned.items()):
        return None
    return tuple(
        _ConstructorArgumentProof(attribute, next(
            proof.type_name for proof in proofs if proof.name == parameter
        ))
        for attribute, parameter in assigned.items()
    )


def _observed_constructor_argument_types(
    tree: ast.Module,
    class_name: str,
    arguments: list[ast.arg],
) -> dict[str, str]:
    observed: dict[str, set[str]] = {item.arg: set() for item in arguments}
    positions = {item.arg: index for index, item in enumerate(arguments)}
    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == class_name
    ):
        for index, value in enumerate(call.args):
            if index >= len(arguments):
                break
            type_name = primitive_literal_type(value)
            if type_name is not None:
                observed[arguments[index].arg].add(type_name)
        for keyword in call.keywords:
            if keyword.arg in positions:
                type_name = primitive_literal_type(keyword.value)
                if type_name is not None:
                    observed[keyword.arg].add(type_name)
    return {
        name: next(iter(types))
        for name, types in observed.items()
        if len(types) == 1
    }


def _method_parameter_types(
    tree: ast.Module,
    method: ast.FunctionDef,
    class_name: str,
) -> dict[str, str]:
    parameters = tuple(item.arg for item in method.args.args[1:])
    inference = infer_primitive_parameter_types(method, set(parameters))
    result: dict[str, str] = {}
    defaults = (None,) * (len(parameters) - len(method.args.defaults)) + tuple(
        method.args.defaults
    )
    for argument, default in zip(method.args.args[1:], defaults):
        type_name = primitive_annotation_type(argument.annotation)
        if type_name is None and default is not None:
            type_name = primitive_literal_type(default)
        if type_name is None:
            type_name = inference.type_for(argument.arg)
        if type_name is not None:
            result[argument.arg] = type_name
    if len(result) != len(parameters):
        observed = _observed_method_argument_types(
            tree,
            class_name=class_name,
            method_name=method.name,
            parameters=parameters,
        )
        for name, type_name in observed.items():
            result.setdefault(name, type_name)
    return result


def _observed_method_argument_types(
    tree: ast.Module,
    *,
    class_name: str,
    method_name: str,
    parameters: tuple[str, ...],
) -> dict[str, str]:
    observed: dict[str, set[str]] = {name: set() for name in parameters}
    scopes: list[tuple[ast.stmt, ...]] = [tuple(tree.body)]
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(tuple(statement.body))
        elif isinstance(statement, ast.ClassDef):
            scopes.extend(
                tuple(member.body)
                for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

    for statements in scopes:
        nodes = _lexical_scope_nodes(statements)
        store_counts: dict[str, int] = {}
        constructors: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                store_counts[node.id] = store_counts.get(node.id, 0) + 1
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == class_name
            ):
                constructors.add(node.targets[0].id)
        proven_receivers = {
            name for name in constructors if store_counts.get(name) == 1
        }
        for call in (node for node in nodes if isinstance(node, ast.Call)):
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == method_name
                and (
                    isinstance(call.func.value, ast.Name)
                    and call.func.value.id in proven_receivers
                    or isinstance(call.func.value, ast.Call)
                    and isinstance(call.func.value.func, ast.Name)
                    and call.func.value.func.id == class_name
                )
            ):
                continue
            for name, value in zip(parameters, call.args):
                type_name = primitive_literal_type(value)
                if type_name is not None:
                    observed[name].add(type_name)
            for keyword in call.keywords:
                if keyword.arg in observed:
                    type_name = primitive_literal_type(keyword.value)
                    if type_name is not None:
                        observed[keyword.arg].add(type_name)
    return {
        name: next(iter(types))
        for name, types in observed.items()
        if len(types) == 1
    }


def _lexical_scope_nodes(
    statements: tuple[ast.stmt, ...],
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []
    pending: list[ast.AST] = list(reversed(statements))
    while pending:
        node = pending.pop()
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            continue
        nodes.append(node)
        pending.extend(reversed(tuple(ast.iter_child_nodes(node))))
    return tuple(nodes)


def _bounded_numeric_observer_attribute(
    method: ast.FunctionDef,
    *,
    collection_attribute: str,
) -> str | None:
    if _observer_has_unsafe_calls(method, collection_attribute):
        return None
    if not _has_bounded_values_loop(method, collection_attribute):
        return None
    candidates: set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        candidates.update(
            item.attr
            for item in ast.walk(node)
            if isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and item.value.id != "self"
        )
    returns_numeric = any(
        isinstance(node, ast.Return)
        and (
            isinstance(node.value, ast.Name)
            or isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "round"
        )
        for node in ast.walk(method)
    )
    return (
        next(iter(candidates))
        if len(candidates) == 1 and returns_numeric
        else None
    )


def _bounded_single_item_observer_attribute(
    method: ast.FunctionDef,
    *,
    collection_attribute: str,
) -> str | None:
    if _observer_has_unsafe_calls(method, collection_attribute):
        return None
    if not _has_bounded_values_loop(method, collection_attribute):
        return None
    candidates = {
        node.value.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id != "self"
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _has_bounded_values_loop(
    method: ast.FunctionDef,
    collection_attribute: str,
) -> bool:
    loops = tuple(node for node in ast.walk(method) if isinstance(node, ast.For))
    return len(loops) == 1 and isinstance(loops[0].iter, ast.Call) and (
        isinstance(loops[0].iter.func, ast.Attribute)
        and loops[0].iter.func.attr == "values"
        and isinstance(loops[0].iter.func.value, ast.Attribute)
        and isinstance(loops[0].iter.func.value.value, ast.Name)
        and loops[0].iter.func.value.value.id == "self"
        and loops[0].iter.func.value.attr == collection_attribute
        and not loops[0].iter.args
        and not loops[0].iter.keywords
    )


def _observer_has_unsafe_calls(
    method: ast.FunctionDef,
    collection_attribute: str,
) -> bool:
    if method.decorator_list or _contains_unsafe_setup_nodes(method):
        return True
    if any(
        isinstance(node, (ast.Attribute, ast.Subscript))
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(method)
    ):
        return True
    for call in (node for node in ast.walk(method) if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Name) and call.func.id == "round":
            continue
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "values"
            and isinstance(call.func.value, ast.Attribute)
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "self"
            and call.func.value.attr == collection_attribute
            and not call.args
            and not call.keywords
        ):
            continue
        return True
    return False


def _is_safe_nested_constructor(class_node: ast.ClassDef) -> bool:
    if class_node.bases or class_node.keywords or class_node.decorator_list:
        return False
    for statement in class_node.body:
        if _is_docstring_statement(statement):
            continue
        if not isinstance(statement, ast.FunctionDef):
            return False
        if (
            statement.name != "__init__"
            and statement.name.startswith("__")
            and statement.name.endswith("__")
        ):
            return False
    constructor = _class_method(class_node, "__init__")
    if constructor is None:
        return False
    arguments = constructor.args
    if (
        arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or len(arguments.args) < 2
    ):
        return False
    if any(
        default is not None and primitive_literal_type(default) is None
        for default in arguments.defaults
    ):
        return False
    parameter_names = {argument.arg for argument in arguments.args[1:]}
    statements = tuple(
        statement
        for index, statement in enumerate(constructor.body)
        if not (index == 0 and _is_docstring_statement(statement))
    )
    assignments: list[tuple[str, str]] = []
    for statement in statements:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and isinstance(statement.targets[0].value, ast.Name)
            and statement.targets[0].value.id == "self"
            and isinstance(statement.value, ast.Name)
            and statement.value.id in parameter_names
        ):
            return False
        assignments.append((statement.targets[0].attr, statement.value.id))
    return (
        len(assignments) == len(parameter_names)
        and len({attribute for attribute, _ in assignments}) == len(assignments)
        and {parameter for _, parameter in assignments} == parameter_names
    )


def _root_setup_calls_are_safe(
    method: ast.FunctionDef,
    *,
    object_parameter: str,
    quantity_parameter: str,
    nested_method_name: str,
) -> bool:
    expected_nested_calls = 0
    for call in (node for node in ast.walk(method) if isinstance(node, ast.Call)):
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == object_parameter
            and call.func.attr == nested_method_name
            and len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.UnaryOp)
            and isinstance(call.args[0].op, ast.USub)
            and isinstance(call.args[0].operand, ast.Name)
            and call.args[0].operand.id == quantity_parameter
        ):
            expected_nested_calls += 1
            continue
        if (
            isinstance(call.func, ast.Name)
            and call.func.id in {"ValueError", "TypeError"}
            and any(
                isinstance(candidate, ast.Raise) and candidate.exc is call
                for candidate in ast.walk(method)
            )
        ):
            continue
        return False
    return expected_nested_calls == 1


def _root_collection_writes_are_bounded(
    method: ast.FunctionDef,
    *,
    collection_attribute: str,
    object_parameter: str,
    quantity_parameter: str,
    key_attribute: str,
) -> bool:
    if any(
        isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
        for node in ast.walk(method)
    ):
        return False
    assignments = tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) for target in node.targets)
    )
    updates = tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Subscript)
    )
    stored_subscripts = tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
    )
    if len(assignments) != 1 or len(updates) != 1 or len(stored_subscripts) != 2:
        return False
    assignment = assignments[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.value, ast.Dict):
        return False
    insertion = assignment.targets[0]
    if not _is_collection_key_subscript(
        insertion,
        collection_attribute=collection_attribute,
        object_parameter=object_parameter,
        key_attribute=key_attribute,
    ):
        return False
    if not _is_exact_owned_record_literal(
        assignment.value,
        object_parameter=object_parameter,
        quantity_parameter=quantity_parameter,
    ):
        return False
    update = updates[0]
    if (
        not isinstance(update.op, ast.Add)
        or not isinstance(update.value, ast.Name)
        or update.value.id != quantity_parameter
        or not isinstance(update.target.value, ast.Subscript)
        or not _is_collection_key_subscript(
            update.target.value,
            collection_attribute=collection_attribute,
            object_parameter=object_parameter,
            key_attribute=key_attribute,
        )
        or not isinstance(update.target.slice, ast.Constant)
        or type(update.target.slice.value) is not str
    ):
        return False
    return True


def _is_exact_owned_record_literal(
    expression: ast.Dict,
    *,
    object_parameter: str,
    quantity_parameter: str,
) -> bool:
    if len(expression.keys) != 2 or len(expression.values) != 2:
        return False
    if any(
        not isinstance(key, ast.Constant) or type(key.value) is not str
        for key in expression.keys
        if key is not None
    ) or any(key is None for key in expression.keys):
        return False
    keys = tuple(key.value for key in expression.keys if isinstance(key, ast.Constant))
    if len(set(keys)) != 2:
        return False
    values = tuple(
        value.id if isinstance(value, ast.Name) else None
        for value in expression.values
    )
    return sorted(values, key=lambda value: value or "") == sorted(
        (object_parameter, quantity_parameter)
    )


def _is_collection_key_subscript(
    node: ast.expr,
    *,
    collection_attribute: str,
    object_parameter: str,
    key_attribute: str,
) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == collection_attribute
        and isinstance(node.slice, ast.Attribute)
        and isinstance(node.slice.value, ast.Name)
        and node.slice.value.id == object_parameter
        and node.slice.attr == key_attribute
    )


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _target_arguments_are_observer_scoped(
    target: ast.FunctionDef,
    expected_occurrences: dict[str, int],
) -> bool:
    actual = {
        name: sum(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
            for node in ast.walk(target)
        )
        for name in expected_occurrences
    }
    return actual == expected_occurrences


def _fresh_analysis_name(prefix: str, occupied: set[str]) -> str:
    candidate = prefix
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _canonical_setup_value(type_name: str) -> int | float | str | bool:
    values: dict[str, int | float | str | bool] = {
        "bool": False,
        "int": 0,
        "float": 0.0,
        "str": "",
    }
    try:
        return values[type_name]
    except KeyError as error:
        _reject(
            SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
            "Canonical setup primitive domain çözümlenemedi.",
        )
        raise AssertionError("unreachable") from error


def _coerce_setup_value(
    value: object,
    type_name: str,
) -> int | float | str | bool:
    expected = _PRIMITIVE_TYPES.get(type_name)
    if expected is None:
        return _canonical_setup_value(type_name)
    if type(value) is expected:
        return value
    if type_name == "float" and type(value) is int:
        return float(value)
    if type_name == "int" and type(value) is float and value.is_integer():
        return int(value)
    if value is None:
        return _canonical_setup_value(type_name)
    _reject(
        SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED,
        "Observer witness constructor primitive domain'iyle uyuşmuyor.",
    )
    raise AssertionError("unreachable")


_UNRESOLVED_SETUP_VALUE = object()


def _evaluate_setup_guard(
    guard: SafeGuardSummary,
    *,
    call: SafeSetupCall,
    receiver: SafeObjectSlot,
    slots: dict[str, SafeObjectSlot],
) -> bool | None:
    value = _evaluate_setup_expression(
        guard.expression_shape,
        call=call,
        receiver=receiver,
        slots=slots,
    )
    return value if type(value) is bool else None


def _evaluate_setup_expression(
    shape: tuple[Any, ...],
    *,
    call: SafeSetupCall,
    receiver: SafeObjectSlot,
    slots: dict[str, SafeObjectSlot],
) -> object:
    if not shape:
        return _UNRESOLVED_SETUP_VALUE
    tag = shape[0]
    if tag == "LITERAL" and len(shape) == 2:
        literal = shape[1]
        if (
            type(literal) is tuple
            and len(literal) == 2
            and literal[0] in _PRIMITIVE_TYPES
            and type(literal[1]) is _PRIMITIVE_TYPES[literal[0]]
        ):
            return literal[1]
        return _UNRESOLVED_SETUP_VALUE
    if tag == "NAME" and len(shape) == 2 and type(shape[1]) is str:
        for argument in call.arguments:
            if (
                argument.parameter_name == shape[1]
                and argument.object_slot_id is None
            ):
                return argument.value
        return _UNRESOLVED_SETUP_VALUE
    if tag == "ARGUMENT" and len(shape) == 2 and type(shape[1]) is int:
        index = shape[1]
        if 0 <= index < len(call.arguments):
            argument = call.arguments[index]
            if argument.object_slot_id is None:
                return argument.value
        return _UNRESOLVED_SETUP_VALUE
    if (
        tag == "OBJECT_CONSTRUCTOR_ARGUMENT"
        and len(shape) == 3
        and type(shape[1]) is int
        and type(shape[2]) is int
    ):
        argument_index, constructor_index = shape[1], shape[2]
        if not 0 <= argument_index < len(call.arguments):
            return _UNRESOLVED_SETUP_VALUE
        slot_id = call.arguments[argument_index].object_slot_id
        slot = slots.get(slot_id or "")
        if (
            slot is None
            or not 0 <= constructor_index < len(slot.blueprint.constructor_arguments)
        ):
            return _UNRESOLVED_SETUP_VALUE
        return slot.blueprint.constructor_arguments[constructor_index][1]
    if (
        tag == "PROVEN_EMPTY_COLLECTION"
        and len(shape) == 2
        and type(shape[1]) is int
        and 0 <= shape[1] < len(receiver.proven_empty_attributes)
    ):
        return ()
    if tag == "UNARY" and len(shape) == 3:
        operand = _evaluate_setup_expression(
            shape[2],
            call=call,
            receiver=receiver,
            slots=slots,
        )
        if operand is _UNRESOLVED_SETUP_VALUE:
            return operand
        if shape[1] == "USUB" and type(operand) in {int, float}:
            return -operand
        if shape[1] == "NOT" and type(operand) is bool:
            return not operand
        return _UNRESOLVED_SETUP_VALUE
    if tag == "BINARY" and len(shape) == 4:
        left = _evaluate_setup_expression(
            shape[2], call=call, receiver=receiver, slots=slots
        )
        right = _evaluate_setup_expression(
            shape[3], call=call, receiver=receiver, slots=slots
        )
        if (
            left is _UNRESOLVED_SETUP_VALUE
            or right is _UNRESOLVED_SETUP_VALUE
            or type(left) not in {int, float}
            or type(right) not in {int, float}
        ):
            return _UNRESOLVED_SETUP_VALUE
        if shape[1] == "ADD":
            return left + right
        if shape[1] == "SUB":
            return left - right
        if shape[1] == "MULT":
            return left * right
        return _UNRESOLVED_SETUP_VALUE
    if tag == "COMPARE" and len(shape) == 4:
        operators = shape[1]
        comparators = shape[3]
        if (
            type(operators) is not tuple
            or len(operators) != 1
            or type(comparators) is not tuple
            or len(comparators) != 1
        ):
            return _UNRESOLVED_SETUP_VALUE
        left = _evaluate_setup_expression(
            shape[2], call=call, receiver=receiver, slots=slots
        )
        right = _evaluate_setup_expression(
            comparators[0], call=call, receiver=receiver, slots=slots
        )
        if left is _UNRESOLVED_SETUP_VALUE or right is _UNRESOLVED_SETUP_VALUE:
            return _UNRESOLVED_SETUP_VALUE
        operator = operators[0]
        if operator in {"<", "LESS_THAN"}:
            return left < right
        if operator in {"<=", "LESS_THAN_OR_EQUAL"}:
            return left <= right
        if operator in {">", "GREATER_THAN"}:
            return left > right
        if operator in {">=", "GREATER_THAN_OR_EQUAL"}:
            return left >= right
        if operator in {"==", "EQUAL"}:
            return left == right
        if operator in {"!=", "NOT_EQUAL"}:
            return left != right
        if operator in {"is", "IS"}:
            return left is right
        if operator in {"is not", "IS_NOT"}:
            return left is not right
        if operator in {"in", "IN"} and isinstance(right, tuple):
            return left in right
        if operator in {"not in", "NOT_IN"} and isinstance(right, tuple):
            return left not in right
    return _UNRESOLVED_SETUP_VALUE


def materialize_safe_object_setup_plan(
    plan: SafeObjectSetupPlan,
    module: ModuleType,
    *,
    include_receiver: bool = False,
) -> dict[str, object] | tuple[dict[str, object], object | None]:
    """Fresh constructor'lar ve statik class method binding ile planı yürütür."""
    if not isinstance(plan, SafeObjectSetupPlan):
        raise TypeError("plan SafeObjectSetupPlan olmalıdır.")
    if not isinstance(module, ModuleType):
        raise TypeError("module ModuleType olmalıdır.")
    if module.__name__ != plan.module_identity:
        _reject(
            SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
            "Runtime module identity setup planıyla eşleşmiyor.",
        )

    instances: dict[str, object] = {}
    for slot in plan.object_slots:
        runtime_class = module.__dict__.get(slot.blueprint.class_name)
        if type(runtime_class) is not type:
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Setup constructor local static class binding'i değildir.",
            )
        assert isinstance(runtime_class, type)
        constructor_values = {
            name: value for name, value in slot.blueprint.constructor_arguments
        }
        instances[slot.slot_id] = runtime_class(**constructor_values)

    for call in plan.setup_calls:
        receiver = instances[call.receiver_slot_id]
        method_name = call.method_summary.receiver.method_identity
        runtime_method = type(receiver).__dict__.get(method_name)
        if type(runtime_method) is not FunctionType:
            _reject(
                SafeSetupPlanRejectionCategory.INVALID_RUNTIME_BINDING,
                "Setup method static instance-function binding'i değildir.",
            )
        assert isinstance(runtime_method, FunctionType)
        arguments = {
            argument.parameter_name: (
                instances[argument.object_slot_id]
                if argument.object_slot_id is not None
                else argument.value
            )
            for argument in call.arguments
        }
        runtime_method(receiver, **arguments)

    arguments = {
        binding.parameter_name: instances[binding.object_slot_id]
        for binding in plan.target_bindings
    }
    if include_receiver:
        return (
            arguments,
            (
                instances[plan.receiver_slot_id]
                if plan.receiver_slot_id is not None
                else None
            ),
        )
    return arguments


def _budget_rejection(name: str, actual: int) -> None:
    _reject(
        SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED,
        f"Safe setup plan {name} bütçesini aşıyor: {actual}",
    )


def _reject(category: SafeSetupPlanRejectionCategory, message: str) -> None:
    raise SafeSetupPlanRejection(category, message)


def _require_identifier(value: str | None, field_name: str) -> None:
    if type(value) is not str or not value.isidentifier():
        raise ValueError(f"{field_name} geçerli identifier olmalıdır.")


def _require_tuple(value: object, field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} tuple olmalıdır.")


def _validate_module_identity(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(not part.isidentifier() for part in value.split("."))
    ):
        raise ValueError("module_identity canonical dotted ad olmalıdır.")


def _validate_qualified_identity(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or any(not part.isidentifier() for part in value.split("."))
    ):
        raise ValueError("target_identity canonical callable identity olmalıdır.")


def _canonical_primitive(
    value: int | float | str | bool | None,
) -> tuple[Any, ...]:
    if value is None:
        return ("none",)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical setup primitive finite olmalıdır.")
        return ("float", value.hex())
    return (type(value).__name__, value)


def _fingerprint(payload: tuple[Any, ...]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
