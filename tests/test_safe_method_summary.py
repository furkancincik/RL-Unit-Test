from __future__ import annotations

import ast
import json
import textwrap
from dataclasses import FrozenInstanceError, is_dataclass, replace
from pathlib import Path

import pytest

import analyzer.safe_method_summary as safe_method_summary
from analyzer.safe_method_summary import (
    DEFAULT_SAFE_METHOD_BUDGETS,
    SafeGuardSummary,
    SafeLocalCallSummary,
    SafeMethodBudgets,
    SafeMethodCategory,
    SafeMethodRejectionCategory,
    SafeMethodSummary,
    SafeMethodSummaryBundle,
    SafePrimitiveDomain,
    SafeReceiverProof,
    SafeReturnSummary,
    SafeStateEffect,
    analyze_safe_method_summaries,
)


def _bundle(
    source: str,
    *,
    module_identity: str = "proof.subject",
    budgets: SafeMethodBudgets = DEFAULT_SAFE_METHOD_BUDGETS,
) -> SafeMethodSummaryBundle:
    return analyze_safe_method_summaries(
        ast.parse(source),
        module_identity=module_identity,
        budgets=budgets,
    )


def _summary(
    source: str,
    target: str,
    *,
    module_identity: str = "proof.subject",
    budgets: SafeMethodBudgets = DEFAULT_SAFE_METHOD_BUDGETS,
) -> SafeMethodSummary:
    return _bundle(
        source,
        module_identity=module_identity,
        budgets=budgets,
    ).summary_for(target)


def _assert_tuple_tree(value: object) -> None:
    assert not isinstance(value, (dict, list, set))
    if isinstance(value, tuple):
        for item in value:
            _assert_tuple_tree(item)


def _effect_source(body: str, *, initializer: str = "0") -> str:
    indented = "\n".join(f"        {line}" for line in body.splitlines())
    return (
        "class Vessel:\n"
        "    def __init__(self):\n"
        f"        self.value = {initializer}\n\n"
        "    def update(self, amount: int = 2):\n"
        f"{indented}\n"
    )


def test_proof_models_are_frozen_and_contain_only_immutable_collections() -> None:
    source = """
class Register:
    def __init__(self):
        self.total = 0

    def increase(self, amount: int) -> None:
        self.total += amount

    def apply(self, amount: int) -> int:
        if amount > 0:
            self.total += 1
            self.increase(amount)
        return self.total
"""
    bundle = _bundle(source)
    summary = bundle.summary_for("Register.apply")
    instances: tuple[object, ...] = (
        bundle,
        summary,
        summary.receiver,
        summary.parameter_domain("amount"),
        summary.guards[0],
        summary.state_effects[0],
        summary.local_calls[0],
        summary.return_summary,
        summary.budgets,
    )

    assert summary.category is SafeMethodCategory.COMPOSED
    assert summary.operation_kinds == (
        "GUARD",
        "STATE_EFFECT",
        "LOCAL_CALL",
        "RETURN",
    )
    for instance in instances:
        assert is_dataclass(instance)
        assert instance.__dataclass_params__.frozen is True
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(instance, "unexpected", True)

    assert isinstance(bundle.summaries, tuple)
    assert isinstance(summary.guards, tuple)
    assert isinstance(summary.state_effects, tuple)
    assert isinstance(summary.local_calls, tuple)
    assert isinstance(summary.parameter_domains, tuple)
    assert isinstance(summary.receiver, SafeReceiverProof)
    assert isinstance(summary.guards[0], SafeGuardSummary)
    assert isinstance(summary.state_effects[0], SafeStateEffect)
    assert isinstance(summary.local_calls[0], SafeLocalCallSummary)
    assert isinstance(summary.return_summary, SafeReturnSummary)
    assert isinstance(summary.parameter_domain("amount"), SafePrimitiveDomain)


def test_proof_models_reject_mutable_container_injection() -> None:
    summary = _summary(
        """
class ImmutableProof:
    def inspect(self, value: int):
        if value > 0:
            return 1
        return 0
""",
        "ImmutableProof.inspect",
    )
    bundle = SafeMethodSummaryBundle(
        module_identity="proof.subject",
        summaries=(summary,),
    )

    with pytest.raises(TypeError, match="guards"):
        replace(summary, guards=[])
    with pytest.raises(TypeError, match="narrowed_domains"):
        replace(summary.guards[0], narrowed_domains=[])
    with pytest.raises(TypeError, match="summaries"):
        replace(bundle, summaries=[])
    with pytest.raises((TypeError, ValueError), match="module_identity"):
        replace(summary.receiver, module_identity=[])


def test_default_budgets_match_the_gate_d_bounded_contract() -> None:
    assert DEFAULT_SAFE_METHOD_BUDGETS == SafeMethodBudgets(
        max_objects=4,
        constructor_depth=1,
        setup_relation_depth=2,
        top_level_setup_calls=2,
        repeated_mutator_calls=2,
        collection_cardinality=2,
        local_call_depth=1,
        guards=3,
        state_effects=2,
        observer_calls=2,
        executable_statements=7,
    )


def test_canonical_payloads_are_deterministic_immutable_tuple_trees() -> None:
    source = _effect_source("self.value += amount")
    first = _summary(source, "Vessel.update")
    second = _summary(source, "Vessel.update")

    _assert_tuple_tree(first.canonical_execution_payload)
    _assert_tuple_tree(first.canonical_semantic_payload)
    assert first.canonical_execution_payload == second.canonical_execution_payload
    assert first.canonical_semantic_payload == second.canonical_semantic_payload
    assert first.execution_fingerprint == second.execution_fingerprint
    assert first.semantic_shape_digest == second.semantic_shape_digest
    json.dumps(first.canonical_execution_payload)
    json.dumps(first.canonical_semantic_payload)


def test_execution_fingerprint_separates_same_shape_across_modules() -> None:
    source = _effect_source("self.value += amount")
    first = _summary(
        source,
        "Vessel.update",
        module_identity="first_package.module",
    )
    second = _summary(
        source,
        "Vessel.update",
        module_identity="second_package.module",
    )

    assert first.execution_fingerprint != second.execution_fingerprint
    assert first.semantic_shape_digest == second.semantic_shape_digest
    assert "first_package.module" in repr(first.canonical_execution_payload)
    assert "second_package.module" in repr(second.canonical_execution_payload)
    assert "first_package.module" not in repr(first.canonical_semantic_payload)


@pytest.mark.parametrize(
    (
        "module_identity",
        "class_name",
        "method_name",
        "parameter_name",
        "attribute_name",
        "local_name",
    ),
    (
        ("renamed.alpha", "Accumulator", "advance", "delta", "level", "next_value"),
        ("renamed.beta", "Gauge", "shift", "step", "reading", "candidate"),
        ("renamed.gamma", "Ledger", "record", "change", "balance", "updated"),
    ),
)
def test_three_fully_renamed_fixtures_share_one_semantic_shape(
    module_identity: str,
    class_name: str,
    method_name: str,
    parameter_name: str,
    attribute_name: str,
    local_name: str,
    request: pytest.FixtureRequest,
) -> None:
    source = f"""
class {class_name}:
    def __init__(self):
        self.{attribute_name} = 0

    def {method_name}(self, {parameter_name}: int) -> int:
        if {parameter_name} > 0:
            {local_name} = {parameter_name} + 1
            self.{attribute_name} += {local_name}
        return self.{attribute_name}
"""
    summary = _summary(
        source,
        f"{class_name}.{method_name}",
        module_identity=module_identity,
    )
    request.node.user_properties.append(
        ("semantic_shape_digest", summary.semantic_shape_digest)
    )

    assert summary.supported is True
    assert summary.category is SafeMethodCategory.COMPOSED
    assert summary.parameter_domain(parameter_name).admissible_types == ("int",)
    assert summary.semantic_shape_digest


def test_three_renamed_fixtures_have_equal_shape_and_distinct_execution_identity() -> None:
    variants = (
        ("rename.one", "Accumulator", "advance", "delta", "level", "next_value"),
        ("rename.two", "Gauge", "shift", "step", "reading", "candidate"),
        ("rename.three", "Ledger", "record", "change", "balance", "updated"),
    )
    summaries: list[SafeMethodSummary] = []
    for module, class_name, method, parameter, attribute, local in variants:
        source = f"""
class {class_name}:
    def __init__(self):
        self.{attribute} = 0

    def {method}(self, {parameter}: int) -> int:
        if {parameter} > 0:
            {local} = {parameter} + 1
            self.{attribute} += {local}
        return self.{attribute}
"""
        summaries.append(
            _summary(source, f"{class_name}.{method}", module_identity=module)
        )

    assert len({item.semantic_shape_digest for item in summaries}) == 1
    assert len({item.execution_fingerprint for item in summaries}) == 3
    assert {item.category for item in summaries} == {SafeMethodCategory.COMPOSED}
    assert len({item.operation_kinds for item in summaries}) == 1
    assert {
        item.parameter_domains[0][1].admissible_types for item in summaries
    } == {("int",)}


@pytest.mark.parametrize(
    ("changed_body", "expected_effect_kind"),
    (
        ("self.value += amount + 3", "AUGMENT_ADD"),
        ("self.value -= amount + 2", "AUGMENT_SUBTRACT"),
    ),
)
def test_literal_or_operator_change_changes_semantic_shape(
    changed_body: str,
    expected_effect_kind: str,
) -> None:
    baseline = _summary(
        _effect_source("self.value += amount + 2"),
        "Vessel.update",
    )
    changed = _summary(_effect_source(changed_body), "Vessel.update")

    assert changed.semantic_shape_digest != baseline.semantic_shape_digest
    assert changed.state_effects[0].kind == expected_effect_kind


@pytest.mark.parametrize(
    ("annotation", "expected"),
    (("int", ("int",)), ("float", ("float",)), ("str", ("str",)), ("bool", ("bool",))),
)
def test_annotation_produces_an_exact_primitive_domain(
    annotation: str,
    expected: tuple[str, ...],
) -> None:
    source = f"""
class Holder:
    def inspect(self, value: {annotation}):
        return value
"""
    summary = _summary(source, "Holder.inspect")

    assert summary.parameter_domain("value").admissible_types == expected
    assert summary.parameter_domain("value").is_exact is True
    assert "ANNOTATION" in summary.parameter_domain("value").evidence_kinds


@pytest.mark.parametrize(
    ("annotation", "expected_rejection"),
    (
        ("CustomValue", SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE),
        ("list[int]", SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN),
        ("int | None", SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN),
    ),
)
def test_explicit_non_primitive_annotation_is_not_overridden_by_other_evidence(
    annotation: str,
    expected_rejection: SafeMethodRejectionCategory,
) -> None:
    summary = _summary(
        f"""
class Annotated:
    def inspect(self, value: {annotation} = 1):
        return value + 2
""",
        "Annotated.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is expected_rejection


def test_string_literal_comparison_proves_string_domain() -> None:
    summary = _summary(
        """
class Holder:
    def inspect(self, candidate):
        return candidate == "ready"
""",
        "Holder.inspect",
    )

    assert summary.supported is True
    assert summary.parameter_domain("candidate").admissible_types == ("str",)
    assert "EXACT_LITERAL_COMPARISON" in summary.parameter_domain(
        "candidate"
    ).evidence_kinds


def test_numeric_operation_keeps_the_full_numeric_domain_without_choosing_a_value() -> None:
    summary = _summary(
        """
class Holder:
    def inspect(self, quantity):
        return quantity * 7
""",
        "Holder.inspect",
    )

    domain = summary.parameter_domain("quantity")
    assert domain.admissible_types == ("int", "float")
    assert domain.is_exact is False
    assert domain.canonical_value is None
    assert "NUMERIC_OPERATION" in domain.evidence_kinds


def test_boolean_context_proves_bool_without_collapsing_to_int() -> None:
    summary = _summary(
        """
class Switch:
    def inspect(self, enabled):
        if enabled:
            return True
        return False
""",
        "Switch.inspect",
    )

    domain = summary.parameter_domain("enabled")
    assert domain.admissible_types == ("bool",)
    assert "int" not in domain.admissible_types


def test_hashable_evidence_remains_a_multi_domain_proof() -> None:
    summary = _summary(
        """
class Index:
    def __init__(self):
        self.entries = {}

    def contains(self, key):
        return key in self.entries
""",
        "Index.contains",
    )

    domain = summary.parameter_domain("key")
    assert domain.admissible_types == ("bool", "int", "float", "str")
    assert domain.is_exact is False
    assert domain.canonical_value is None
    assert domain.allows_nan is False
    assert domain.allows_infinity is False


def test_conflicting_numeric_and_string_evidence_is_controlled_unsupported() -> None:
    summary = _summary(
        """
class Inspector:
    def classify(self, value):
        if value < 9:
            return 1
        return value == "nine"
""",
        "Inspector.classify",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.supported is False
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_bool_and_int_defaults_produce_distinct_domains_and_digests() -> None:
    source = """
class Defaults:
    def flag(self, value=False):
        return value

    def number(self, value=0):
        return value
"""
    bundle = _bundle(source)
    flag = bundle.summary_for("Defaults.flag")
    number = bundle.summary_for("Defaults.number")

    assert flag.parameter_domain("value").admissible_types == ("bool",)
    assert number.parameter_domain("value").admissible_types == ("int",)
    assert flag.semantic_shape_digest != number.semantic_shape_digest


def test_direct_assignment_does_not_invent_a_parameter_type() -> None:
    summary = _summary(
        """
class Holder:
    def __init__(self):
        self.value = 0

    def replace(self, incoming):
        self.value = incoming
""",
        "Holder.replace",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
    )


def test_unused_untyped_parameter_is_not_promoted_to_a_safe_primitive() -> None:
    summary = _summary(
        """
class ConstantObserver:
    def observe(self, unknown):
        return 1
""",
        "ConstantObserver.observe",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
    )


def test_non_receiver_attribute_access_is_not_a_safe_primitive_operation() -> None:
    summary = _summary(
        """
class PrimitiveObserver:
    def observe(self, value: int):
        if value.real:
            return 1
        return 0
""",
        "PrimitiveObserver.observe",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNKNOWN_OBJECT_GRAPH
    )


@pytest.mark.parametrize("parameter_name", ("amount", "delta", "increment"))
def test_parameter_names_do_not_change_numeric_domain(parameter_name: str) -> None:
    summary = _summary(
        f"""
class Meter:
    def calculate(self, {parameter_name}):
        return {parameter_name} + 11
""",
        "Meter.calculate",
    )

    assert summary.parameter_domain(parameter_name).admissible_types == (
        "int",
        "float",
    )


@pytest.mark.parametrize(
    ("body", "initializer", "expected_kind"),
    (
        ("self.value = 5", "0", "ASSIGN_LITERAL"),
        ("self.value = amount", "0", "ASSIGN_PARAMETER"),
        ("self.value += amount", "0", "AUGMENT_ADD"),
        ("self.value -= amount", "0", "AUGMENT_SUBTRACT"),
        ("self.value.append(amount)", "[]", "LIST_APPEND"),
    ),
)
def test_safe_mutator_effects_are_classified_in_source_order(
    body: str,
    initializer: str,
    expected_kind: str,
) -> None:
    summary = _summary(
        _effect_source(body, initializer=initializer),
        "Vessel.update",
    )

    assert summary.supported is True
    assert summary.category is SafeMethodCategory.MUTATOR
    assert tuple(effect.kind for effect in summary.state_effects) == (expected_kind,)


def test_constructor_proven_empty_dict_accepts_direct_key_assignment() -> None:
    summary = _summary(
        """
class Catalog:
    def __init__(self):
        self.entries = {}

    def insert(self, key: str, value: int) -> None:
        self.entries[key] = value
""",
        "Catalog.insert",
    )

    assert summary.supported is True
    assert summary.category is SafeMethodCategory.MUTATOR
    assert summary.state_effects[0].kind == "DICT_SET"


def test_set_add_stays_unsupported_without_constructor_set_provenance() -> None:
    summary = _summary(
        """
class Registry:
    def __init__(self):
        self.values = set()

    def add(self, value: int) -> None:
        self.values.add(value)
""",
        "Registry.add",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category in {
        SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE,
        SafeMethodRejectionCategory.UNSAFE_CALL,
    }


@pytest.mark.parametrize(
    "body",
    ("return self.values[0]", "return self.mapping['missing']"),
)
def test_constructor_empty_collection_lookup_requires_presence_proof(body: str) -> None:
    summary = _summary(
        f"""
class EmptyState:
    def __init__(self):
        self.values = []
        self.mapping = {{}}

    def inspect(self):
        {body}
""",
        "EmptyState.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNPROVEN_LOOKUP


@pytest.mark.parametrize(
    ("body", "expected_kind"),
    (
        ("return self.value", "ATTRIBUTE_RETURN"),
        ("return bool(self.values)", "COLLECTION_TRUTHINESS"),
        ("return item in self.values", "MEMBERSHIP"),
        ("return len(self.values)", "LENGTH"),
        ("return self.value + offset", "ARITHMETIC"),
    ),
)
def test_safe_observer_return_shapes_are_explicit(
    body: str,
    expected_kind: str,
) -> None:
    source = f"""
class View:
    def __init__(self):
        self.value = 2
        self.values = []

    def inspect(self, item: int = 1, offset: int = 3):
        {body}
"""
    summary = _summary(source, "View.inspect")

    assert summary.supported is True
    assert summary.category is SafeMethodCategory.OBSERVER
    assert summary.return_summary.kind == expected_kind


def test_guard_summary_tracks_kind_operator_polarity_and_narrowed_domain() -> None:
    summary = _summary(
        """
class Gate:
    def inspect(self, amount):
        if amount >= 4:
            return "upper"
        return "lower"
""",
        "Gate.inspect",
    )

    guard = summary.guards[0]
    assert guard.kind == "PRIMITIVE_COMPARISON"
    assert guard.operator == "GREATER_THAN_OR_EQUAL"
    assert guard.branch_polarities == (True, False)
    assert guard.narrowed_domains == (("parameter:amount", ("int", "float")),)


def test_composed_summary_preserves_safe_local_call_closure_without_execution() -> None:
    summary = _summary(
        """
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self, amount: int) -> None:
        self.value += amount

    def update(self, amount: int) -> int:
        if amount > 0:
            self.value += 1
            self.increment(amount)
        return self.value
""",
        "Counter.update",
    )

    assert summary.supported is True
    assert summary.category is SafeMethodCategory.COMPOSED
    assert tuple(call.method_identity for call in summary.local_calls) == (
        "increment",
    )
    assert summary.local_calls[0].call_depth == 1
    assert summary.local_calls[0].callee_execution_fingerprint
    assert summary.operation_kinds == (
        "GUARD",
        "STATE_EFFECT",
        "LOCAL_CALL",
        "RETURN",
    )


def test_renamed_local_call_return_targets_share_one_semantic_shape() -> None:
    variants = (
        ("calls.first", "Counter", "increase", "apply", "amount", "value"),
        ("calls.second", "Gauge", "shift", "update", "step", "reading"),
        ("calls.third", "Ledger", "record", "commit", "change", "balance"),
    )
    summaries: list[SafeMethodSummary] = []
    for module, class_name, callee, caller, parameter, attribute in variants:
        summaries.append(
            _summary(
                f"""
class {class_name}:
    def __init__(self):
        self.{attribute} = 0

    def {callee}(self, {parameter}: int) -> None:
        self.{attribute} += {parameter}

    def {caller}(self, {parameter}: int):
        return self.{callee}({parameter})
""",
                f"{class_name}.{caller}",
                module_identity=module,
            )
        )

    assert {summary.category for summary in summaries} == {
        SafeMethodCategory.COMPOSED
    }
    assert len({summary.semantic_shape_digest for summary in summaries}) == 1
    assert len({summary.execution_fingerprint for summary in summaries}) == 3


def test_bounded_fold_return_ir_preserves_all_static_proof_parts() -> None:
    numeric = SafePrimitiveDomain(
        admissible_types=("int",),
        evidence_kinds=("ANNOTATION",),
    )
    guard = SafeGuardSummary(
        kind="PRIMITIVE_COMPARISON",
        operator="GREATER_THAN",
        branch_polarities=(True, False),
        narrowed_domains=(("element.amount", ("int",)),),
    )
    summary = SafeReturnSummary(
        kind="BOUNDED_FOLD",
        domain=numeric,
        source_collection_attribute="entries",
        element_primitive_fields=(("amount", numeric),),
        aggregate_operator="ADD",
        aggregate_initial_value=0,
        aggregate_filter=guard,
        bounded_cardinality_required=2,
    )

    assert summary.source_collection_attribute == "entries"
    assert summary.element_primitive_fields == (("amount", numeric),)
    assert summary.aggregate_operator == "ADD"
    assert summary.aggregate_initial_value == 0
    assert summary.aggregate_filter is guard
    assert summary.bounded_cardinality_required == 2
    _assert_tuple_tree(summary.canonical_execution_payload)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        summary.aggregate_operator = "MULTIPLY"  # type: ignore[misc]


def test_recursive_local_call_is_cycle_safe_and_controlled_unsupported() -> None:
    summary = _summary(
        """
class Cycle:
    def first(self, value: int) -> int:
        return self.second(value)

    def second(self, value: int) -> int:
        return self.first(value)
""",
        "Cycle.first",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.CALL_CYCLE


@pytest.mark.parametrize(
    ("class_prefix", "method", "expected_rejection"),
    (
        ("", "return open('data.txt').read()", "UNSAFE_CALL"),
        ("", "return __import__('socket').socket()", "DYNAMIC_IMPORT"),
        ("", "import subprocess\nreturn subprocess.run(['tool'])", "IMPORT_OR_PROCESS"),
        ("", "global shared\nshared = value\nreturn shared", "GLOBAL_MUTATION"),
        ("", "return helper(value)", "UNSAFE_CALL"),
        ("", "return getattr(self, 'value')", "REFLECTION"),
        ("", "alias = self\nalias.value = value", "UNSAFE_RECEIVER"),
        ("", "del self.value", "DELETE"),
        ("", "try:\n    self.value = value\nfinally:\n    self.value = 0", "EXCEPTION_DEPENDENT_EFFECT"),
        ("", "while True:\n    self.value += 1", "UNBOUNDED_CONTROL_FLOW"),
        ("", "def nested():\n    self.value = value\nreturn nested", "NESTED_SCOPE_MUTATION"),
        ("", "self.child.value = value", "UNKNOWN_OBJECT_GRAPH"),
        ("", "return callback(value)", "CALLBACK"),
        ("", "yield value", "GENERATOR"),
        ("", "with manager:\n    return value", "CONTEXT_MANAGER"),
        ("@property\n    ", "return self.value", "DECORATED_METHOD"),
        ("@staticmethod\n    ", "return value", "STATIC_OR_CLASS_METHOD"),
        ("@classmethod\n    ", "return value", "STATIC_OR_CLASS_METHOD"),
    ),
)
def test_known_unsafe_shapes_return_internal_controlled_categories(
    class_prefix: str,
    method: str,
    expected_rejection: str,
) -> None:
    receiver = "self, value: int" if "staticmethod" not in class_prefix else "value: int"
    if "classmethod" in class_prefix:
        receiver = "cls, value: int"
    if expected_rejection == "CALLBACK":
        receiver = "self, value: int, callback"
    source = (
        "class Unsafe:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n\n"
        f"    {class_prefix}def act({receiver}):\n"
        + textwrap.indent(textwrap.dedent(method), "        ")
        + "\n"
    )
    summary = _summary(source, "Unsafe.act")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.supported is False
    assert summary.rejection_category.name == expected_rejection
    assert summary.internal_rejection_detail is None


@pytest.mark.parametrize(
    ("class_header", "class_name", "method_body", "expected_rejection"),
    (
        (
            "class Child(Base):",
            "Child",
            "return super().act(value)",
            "INHERITANCE_OR_METACLASS",
        ),
        (
            "class Meta(metaclass=Factory):",
            "Meta",
            "return value",
            "INHERITANCE_OR_METACLASS",
        ),
        ("class AsyncTarget:", "AsyncTarget", "return value", "ASYNC_METHOD"),
    ),
)
def test_unsafe_class_and_async_shapes_are_rejected(
    class_header: str,
    class_name: str,
    method_body: str,
    expected_rejection: str,
) -> None:
    keyword = "async def" if class_header == "class AsyncTarget:" else "def"
    source = (
        f"{class_header}\n"
        f"    {keyword} act(self, value: int):\n"
        f"        {method_body}\n"
    )
    summary = _summary(source, f"{class_name}.act")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == expected_rejection


def test_excess_guards_are_not_truncated_and_fail_the_budget() -> None:
    summary = _summary(
        """
class Guarded:
    def inspect(self, value: int) -> int:
        if value > 1:
            return 1
        if value > 2:
            return 2
        if value > 3:
            return 3
        if value > 4:
            return 4
        return 0
""",
        "Guarded.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED
    assert len(summary.guards) == 4


def test_guard_budget_is_rejected_before_exponential_path_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_enumerated(*args: object, **kwargs: object) -> int:
        raise AssertionError("over-budget guards must not enumerate path masks")

    monkeypatch.setattr(
        safe_method_summary,
        "_maximum_guarded_weight",
        fail_if_enumerated,
    )
    summary = _summary(
        """
class GuardLimit:
    def __init__(self):
        self.values = []

    def inspect(self, value: int) -> int:
        if value > 1:
            self.values.append(value)
            return 1
        if value > 2:
            return 2
        if value > 3:
            return 3
        if value > 4:
            return 4
        return 0
""",
        "GuardLimit.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_excess_state_effects_are_not_truncated_and_fail_the_budget() -> None:
    summary = _summary(
        """
class Multi:
    def __init__(self):
        self.first = 0
        self.second = 0
        self.third = 0

    def update(self) -> None:
        self.first = 1
        self.second = 2
        self.third = 3
""",
        "Multi.update",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED
    assert len(summary.state_effects) == 3


def test_local_call_depth_budget_rejects_a_transitive_chain() -> None:
    summary = _summary(
        """
class Chain:
    def __init__(self):
        self.value = 0

    def tail(self, value: int) -> None:
        self.value += value

    def middle(self, value: int) -> None:
        self.tail(value)

    def head(self, value: int) -> None:
        self.middle(value)
""",
        "Chain.head",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ("return [item for item in self.values]", "UNSAFE_CONSTRUCT"),
        ("if (current := value):\n    return current\nreturn 0", "UNSAFE_CONSTRUCT"),
        ("return self.missing(value)", "UNKNOWN_METHOD"),
        ("return self.safe(*values)", "UNKNOWN_PRIMITIVE_DOMAIN"),
        ("return self.safe(value=value)", "UNSAFE_CALL"),
    ),
)
def test_dynamic_guard_and_call_shapes_are_controlled_unsupported(
    body: str,
    expected: str,
) -> None:
    source = (
        "class Dynamic:\n"
        "    def __init__(self):\n"
        "        self.values = []\n\n"
        "    def safe(self, value: int):\n"
        "        return value\n\n"
        "    def inspect(self, value: int, values):\n"
        + textwrap.indent(body, "        ")
        + "\n"
    )
    summary = _summary(source, "Dynamic.inspect")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == expected


def test_local_call_argument_domain_must_match_callee_signature() -> None:
    summary = _summary(
        """
class LocalDomain:
    def consume(self, value: str) -> str:
        return value

    def inspect(self, value: int):
        return self.consume(value)
""",
        "LocalDomain.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_classifier_does_not_execute_import_constructor_or_method_code(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must_not_exist.txt"
    source = f"""
__import__('pathlib').Path({str(marker)!r}).write_text('module', encoding='utf-8')

class Dangerous:
    def __init__(self):
        __import__('pathlib').Path({str(marker)!r}).write_text('constructor', encoding='utf-8')

    def act(self):
        __import__('pathlib').Path({str(marker)!r}).write_text('method', encoding='utf-8')
"""
    summary = _summary(source, "Dangerous.act")

    assert marker.exists() is False
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category in {
        SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE,
        SafeMethodRejectionCategory.DYNAMIC_IMPORT,
    }
    assert str(marker) not in repr(summary)
    assert str(marker) not in repr(summary.canonical_execution_payload)
    assert str(marker) not in repr(summary.canonical_semantic_payload)


def test_summary_contains_no_raw_source_ast_path_or_runtime_identity() -> None:
    source = _effect_source("self.value += amount")
    summary = _summary(source, "Vessel.update", module_identity="safe.module")
    combined = repr(
        (
            summary,
            summary.canonical_execution_payload,
            summary.canonical_semantic_payload,
        )
    )

    assert source not in combined
    assert "<ast." not in combined
    assert "0x" not in combined
    assert "lineno" not in combined
    assert "path_index" not in combined
    assert "scenario" not in combined.lower()


def test_bundle_order_is_source_deterministic() -> None:
    bundle = _bundle(
        """
class First:
    def beta(self, value: int):
        return value

    def alpha(self, value: int):
        return value

class Second:
    def gamma(self, value: int):
        return value
"""
    )

    assert tuple(summary.qualified_name for summary in bundle.summaries) == (
        "First.beta",
        "First.alpha",
        "Second.gamma",
    )


def test_unknown_summary_target_uses_a_domain_error_not_a_mapping_key_error() -> None:
    bundle = _bundle(
        """
class Present:
    def inspect(self, value: int):
        return value
"""
    )

    with pytest.raises(ValueError, match="summary"):
        bundle.summary_for("Missing.inspect")


def test_local_binding_expression_is_part_of_both_method_identities() -> None:
    def summarized(literal: int) -> SafeMethodSummary:
        return _summary(
            f"""
class Accumulator:
    def __init__(self):
        self.total = 0

    def increase(self, amount: int) -> int:
        computed = amount + {literal}
        self.total += computed
        return self.total
""",
            "Accumulator.increase",
        )

    first = summarized(1)
    second = summarized(2)

    assert first.supported is True
    assert second.supported is True
    assert first.execution_fingerprint != second.execution_fingerprint
    assert first.semantic_shape_digest != second.semantic_shape_digest


def test_conditional_effects_and_calls_retain_guard_polarity_context() -> None:
    summary = _summary(
        """
class Conditional:
    def __init__(self):
        self.value = 0

    def apply(self, amount: int) -> None:
        self.value += amount

    def update(self, amount: int, enabled: bool) -> int:
        if enabled:
            self.value = amount
            self.apply(amount)
        else:
            self.value = 0
        return self.value
""",
        "Conditional.update",
    )

    assert summary.supported is True
    assert tuple(effect.guard_context for effect in summary.state_effects) == (
        ((0, True),),
        ((0, False),),
    )
    assert tuple(call.guard_context for call in summary.local_calls) == (
        ((0, True),),
    )
    assert summary.state_effects[0].canonical_execution_payload != (
        summary.state_effects[1].canonical_execution_payload
    )


def test_early_return_guard_polarity_is_carried_into_fallthrough_effects() -> None:
    summary = _summary(
        """
class Fallthrough:
    def __init__(self):
        self.first = 0
        self.second = 0
        self.third = 0
        self.fourth = 0

    def update(self, enabled: bool):
        if enabled:
            self.first = 1
            self.second = 2
            return
        self.third = 3
        self.fourth = 4
""",
        "Fallthrough.update",
    )

    assert summary.supported is True
    assert tuple(effect.guard_context for effect in summary.state_effects) == (
        ((0, True),),
        ((0, True),),
        ((0, False),),
        ((0, False),),
    )


def test_disjunctive_nested_fallthrough_is_controlled_unsupported() -> None:
    summary = _summary(
        """
class DisjunctiveFallthrough:
    def __init__(self):
        self.first = 0
        self.second = 0
        self.third = 0
        self.fourth = 0

    def update(self, outer: bool, inner: bool):
        if outer:
            if inner:
                self.first = 1
                self.second = 2
                return
        self.third = 3
        self.fourth = 4
""",
        "DisjunctiveFallthrough.update",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


@pytest.mark.parametrize(
    "body",
    (
        "self.value = replacement",
        "self.value = 'wrong'\nself.value += 1",
    ),
)
def test_ordered_state_effects_preserve_domain_compatibility(body: str) -> None:
    source = (
        "class State:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n\n"
        "    def update(self, replacement: str) -> None:\n"
        + textwrap.indent(body, "        ")
        + "\n"
    )

    summary = _summary(source, "State.update")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_local_call_requires_actual_domain_to_be_a_safe_subset() -> None:
    summary = _summary(
        """
class Consumer:
    def consume(self, value: int) -> int:
        return value

    def inspect(self, amount):
        numeric_evidence = amount * 2
        return self.consume(amount)
""",
        "Consumer.inspect",
    )

    assert summary.parameter_domain("amount").admissible_types == ("int", "float")
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_local_call_literal_and_expression_are_part_of_method_identities() -> None:
    def summarized(argument: str) -> SafeMethodSummary:
        return _summary(
            f"""
class Forwarder:
    def consume(self, value: int) -> int:
        return value

    def forward(self, value: int) -> int:
        return self.consume({argument})
""",
            "Forwarder.forward",
        )

    literal_one = summarized("1")
    literal_two = summarized("2")
    expression_one = summarized("value + 1")
    expression_two = summarized("value + 2")

    assert literal_one.execution_fingerprint != literal_two.execution_fingerprint
    assert literal_one.semantic_shape_digest != literal_two.semantic_shape_digest
    assert expression_one.execution_fingerprint != expression_two.execution_fingerprint
    assert expression_one.semantic_shape_digest != expression_two.semantic_shape_digest


@pytest.mark.parametrize(
    "signature",
    (
        "self, *, value: int = 1",
        "self, *values: int",
        "self, **values: int",
        "self, value: int, /",
    ),
)
def test_non_core_method_signatures_are_controlled_unsupported(
    signature: str,
) -> None:
    summary = _summary(
        f"""
class Signature:
    def inspect({signature}) -> int:
        return 1
""",
        "Signature.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == "UNSAFE_SIGNATURE"


@pytest.mark.parametrize(
    "signature",
    (
        "self, *, value: int = 1",
        "self, *values: int",
        "self, **values: int",
    ),
)
def test_non_core_constructor_signatures_do_not_create_receiver_proof(
    signature: str,
) -> None:
    summary = _summary(
        f"""
class Constructed:
    def __init__({signature}):
        self.value = 0

    def inspect(self) -> int:
        return self.value
""",
        "Constructed.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE
    )


def test_untyped_required_constructor_parameter_does_not_create_receiver_proof() -> None:
    summary = _summary(
        """
class Constructed:
    def __init__(self, unknown):
        self.value = 0

    def inspect(self) -> int:
        return self.value
""",
        "Constructed.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE
    )


@pytest.mark.parametrize(
    "body",
    (
        "alias = self.values\nreturn 0",
        "self = replacement\nreturn 0",
    ),
)
def test_collection_alias_and_self_rebinding_are_controlled_unsupported(
    body: str,
) -> None:
    source = (
        "class Alias:\n"
        "    def __init__(self):\n"
        "        self.values = []\n\n"
        "    def inspect(self, replacement: int) -> int:\n"
        + textwrap.indent(body, "        ")
        + "\n"
    )

    summary = _summary(source, "Alias.inspect")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_RECEIVER


def test_deep_expression_tree_exceeds_a_bounded_static_analysis_budget() -> None:
    expression = " + ".join(["value"] * 80)
    summary = _summary(
        f"""
class Bounded:
    def inspect(self, value: int) -> int:
        return {expression}
""",
        "Bounded.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ("self.values.append(len(self.missing))", "UNKNOWN_OBJECT_GRAPH"),
        ("return self.consume(len(self.missing))", "UNKNOWN_OBJECT_GRAPH"),
        ("return self.consume(len(self.values, self.values))", "UNSAFE_CALL"),
    ),
)
def test_nested_call_arguments_are_validated_before_effect_or_local_call_summary(
    body: str,
    expected: str,
) -> None:
    summary = _summary(
        "class NestedArgument:\n"
        "    def __init__(self):\n"
        "        self.values = []\n\n"
        "    def consume(self, value: int) -> int:\n"
        "        return value\n\n"
        "    def inspect(self):\n"
        + textwrap.indent(body, "        ")
        + "\n",
        "NestedArgument.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == expected


def test_parameter_rebinding_is_not_treated_as_a_primitive_local_binding() -> None:
    summary = _summary(
        """
class Rebinding:
    def inspect(self, value: int):
        value = "changed"
        return value
""",
        "Rebinding.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_branch_local_binding_is_not_available_outside_its_guard_context() -> None:
    summary = _summary(
        """
class BranchLocal:
    def inspect(self, enabled: bool):
        if enabled:
            temporary = 1
        return temporary
""",
        "BranchLocal.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
    )


@pytest.mark.parametrize(
    "body",
    (
        "enabled and self.values.append(value)",
        "self.consume(value) if enabled else self.consume(0)",
    ),
)
def test_conditional_effect_or_call_expression_is_controlled_unsupported(
    body: str,
) -> None:
    summary = _summary(
        "class Conditional:\n"
        "    def __init__(self):\n"
        "        self.values = []\n\n"
        "    def consume(self, value: int) -> int:\n"
        "        return value\n\n"
        "    def inspect(self, enabled: bool, value: int):\n"
        + textwrap.indent(body, "        ")
        + "\n        return 0\n",
        "Conditional.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_ordering_against_a_bool_literal_is_a_domain_conflict() -> None:
    summary = _summary(
        """
class Ordering:
    def inspect(self, value):
        return value < True
""",
        "Ordering.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


@pytest.mark.parametrize(
    "body",
    (
        "if MISSING:\n    return 1\nreturn 0",
        "temporary = MISSING\nreturn 1",
        "return bool(MISSING)",
    ),
)
def test_unknown_name_provenance_is_controlled_unsupported(body: str) -> None:
    summary = _summary(
        "class UnknownName:\n"
        "    def inspect(self):\n"
        + textwrap.indent(body, "        ")
        + "\n",
        "UnknownName.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNKNOWN_PRIMITIVE_DOMAIN
    )


def test_len_requires_constructor_proven_collection_state() -> None:
    summary = _summary(
        """
class PrimitiveLength:
    def __init__(self):
        self.value = 1

    def inspect(self):
        return len(self.value)
""",
        "PrimitiveLength.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


@pytest.mark.parametrize(
    "expression",
    ("left < right", "needle in haystack"),
)
def test_incompatible_comparison_domains_are_controlled_unsupported(
    expression: str,
) -> None:
    summary = _summary(
        f"""
class Compatibility:
    def inspect(self, left: int, right: str, needle: int, haystack: str):
        return {expression}
""",
        "Compatibility.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_non_call_expression_statement_is_not_silently_ignored() -> None:
    summary = _summary(
        """
class ExpressionStatement:
    def inspect(self):
        1 / 0
        return 1
""",
        "ExpressionStatement.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_constructor_signature_participates_in_receiver_execution_identity() -> None:
    without_parameter = _summary(
        """
class Receiver:
    def __init__(self):
        pass

    def inspect(self):
        return 1
""",
        "Receiver.inspect",
        module_identity="identity.receiver",
    )
    with_parameter = _summary(
        """
class Receiver:
    def __init__(self, seed: int):
        pass

    def inspect(self):
        return 1
""",
        "Receiver.inspect",
        module_identity="identity.receiver",
    )

    assert without_parameter.receiver.class_fingerprint != (
        with_parameter.receiver.class_fingerprint
    )
    assert without_parameter.execution_fingerprint != with_parameter.execution_fingerprint


def test_numeric_return_domain_is_a_safe_overapproximation() -> None:
    multiplied = _summary(
        """
class Numeric:
    def inspect(self, amount):
        return amount * 2
""",
        "Numeric.inspect",
    )
    divided = _summary(
        """
class Numeric:
    def inspect(self, amount: int):
        return amount / 2
""",
        "Numeric.inspect",
    )

    assert multiplied.return_summary.domain is not None
    assert multiplied.return_summary.domain.admissible_types == ("int", "float")
    assert divided.return_summary.domain is not None
    assert divided.return_summary.domain.admissible_types == ("float",)


def test_primitive_bool_call_has_an_explicit_safe_return_summary() -> None:
    summary = _summary(
        """
class BooleanObserver:
    def inspect(self, value: int):
        return bool(value)
""",
        "BooleanObserver.inspect",
    )

    assert summary.category is SafeMethodCategory.OBSERVER
    assert summary.return_summary.kind == "BOOLEAN_CONTEXT"
    assert summary.return_summary.domain is not None
    assert summary.return_summary.domain.admissible_types == ("bool",)


@pytest.mark.parametrize("operator", ("/", "//", "%"))
def test_literal_zero_divisor_is_controlled_unsupported(operator: str) -> None:
    summary = _summary(
        f"""
class ZeroDivisor:
    def inspect(self, value: int):
        return value {operator} 0
""",
        "ZeroDivisor.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_short_circuit_local_call_inside_guard_is_controlled_unsupported() -> None:
    summary = _summary(
        """
class GuardCall:
    def consume(self, value: int) -> bool:
        return bool(value)

    def inspect(self, enabled: bool, value: int):
        if enabled and self.consume(value):
            return 1
        return 0
""",
        "GuardCall.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("left + right", "DOMAIN_CONFLICT"),
        ("+right", "DOMAIN_CONFLICT"),
        ("right[0]", "UNPROVEN_LOOKUP"),
        ("left << left", "DOMAIN_CONFLICT"),
        ("left @ left", "DOMAIN_CONFLICT"),
    ),
)
def test_guard_expression_grammar_rejects_unproven_or_incompatible_operations(
    expression: str,
    expected: str,
) -> None:
    summary = _summary(
        f"""
class GuardGrammar:
    def inspect(self, left: int, right: str):
        if {expression}:
            return 1
        return 0
""",
        "GuardGrammar.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == expected


def test_value_return_on_only_some_paths_is_controlled_unsupported() -> None:
    summary = _summary(
        """
class PartialReturn:
    def inspect(self, enabled: bool):
        if enabled:
            return 1
""",
        "PartialReturn.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_unreachable_effect_after_return_is_not_summarized() -> None:
    summary = _summary(
        """
class DeadEffect:
    def __init__(self):
        self.value = 0

    def inspect(self, amount: int):
        return 1
        self.value += amount
""",
        "DeadEffect.inspect",
    )

    assert summary.category is SafeMethodCategory.OBSERVER
    assert summary.state_effects == ()
    assert summary.operation_kinds == ("RETURN",)


def test_returned_safe_collection_mutator_has_exact_none_return_proof() -> None:
    summary = _summary(
        """
class CollectionMutator:
    def __init__(self):
        self.values = []

    def add(self, value: int):
        return self.values.append(value)
""",
        "CollectionMutator.add",
    )

    assert summary.category is SafeMethodCategory.MUTATOR
    assert summary.return_summary.kind == "NONE"
    assert tuple(effect.kind for effect in summary.state_effects) == ("LIST_APPEND",)


@pytest.mark.parametrize(
    "guard",
    ("self.values.append(value)", "self.consume(value)"),
)
def test_effectful_or_local_method_call_is_not_a_safe_guard(guard: str) -> None:
    summary = _summary(
        f"""
class GuardMethodCall:
    def __init__(self):
        self.values = []

    def consume(self, value: int) -> bool:
        return bool(value)

    def inspect(self, value: int):
        if {guard}:
            return 1
        return 0
""",
        "GuardMethodCall.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_mutated_state_cannot_reuse_constructor_nonzero_divisor_proof() -> None:
    summary = _summary(
        """
class Divider:
    def __init__(self):
        self.divisor = 2

    def inspect(self, value: int):
        self.divisor = 0
        return value / self.divisor
""",
        "Divider.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_mutated_state_return_uses_the_current_abstract_domain() -> None:
    summary = _summary(
        """
class StateDomain:
    def __init__(self):
        self.value = 0

    def inspect(self, amount: float):
        self.value += amount
        return self.value
""",
        "StateDomain.inspect",
    )

    assert summary.category is SafeMethodCategory.COMPOSED
    assert summary.return_summary.domain is not None
    assert summary.return_summary.domain.admissible_types == ("int", "float")


def test_parameter_default_is_not_a_nonzero_divisor_proof() -> None:
    summary = _summary(
        """
class ParameterDivider:
    def inspect(self, value: int, divisor=2):
        return value / divisor
""",
        "ParameterDivider.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_branch_state_domains_merge_monotonically_across_alternatives() -> None:
    summary = _summary(
        """
class BranchState:
    def __init__(self):
        self.value = 0

    def inspect(self, enabled: bool, amount: float):
        if enabled:
            self.value += amount
        else:
            self.value = 1
        return self.value
""",
        "BranchState.inspect",
    )

    assert summary.category is SafeMethodCategory.COMPOSED
    assert summary.return_summary.domain is not None
    assert summary.return_summary.domain.admissible_types == ("int", "float")


@pytest.mark.parametrize("builtin_name", ("bool", "len"))
def test_local_binding_cannot_be_misclassified_as_a_safe_builtin(
    builtin_name: str,
) -> None:
    summary = _summary(
        f"""
class LocalShadow:
    def inspect(self, value: int):
        {builtin_name} = value
        return {builtin_name}(value)
""",
        "LocalShadow.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CALL


@pytest.mark.parametrize("builtin_name", ("bool", "len"))
def test_later_local_binding_shadows_builtin_for_the_entire_method_scope(
    builtin_name: str,
) -> None:
    summary = _summary(
        f"""
class LexicalShadow:
    def inspect(self, value: int):
        result = {builtin_name}(value)
        {builtin_name} = value
        return result
""",
        "LexicalShadow.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CALL


@pytest.mark.parametrize("builtin_name", ("bool", "len"))
def test_module_binding_cannot_be_misclassified_as_a_safe_builtin(
    builtin_name: str,
) -> None:
    summary = _summary(
        f"""
def {builtin_name}(value):
    return value

class ModuleShadow:
    def inspect(self, value: int):
        return {builtin_name}(value)
""",
        "ModuleShadow.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CALL


def test_instance_state_cannot_shadow_a_safe_local_method_target() -> None:
    summary = _summary(
        """
class MethodShadow:
    def __init__(self):
        self.consume = 0

    def consume(self, value: int) -> int:
        return value

    def inspect(self, value: int):
        return self.consume(value)
""",
        "MethodShadow.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE
    )


@pytest.mark.parametrize(
    "class_body",
    (
        "__slots__ = ()\n\ndef __init__(self):\n    self.value = 1",
        "def __init__(self):\n    self.value = 1\n\n"
        "def __init__(self):\n    self.value = build_value()",
        "def __init__(self):\n    self.value = 1\n\nvalue = 0",
        "def __setattr__(self, name, value):\n    raise RuntimeError()\n\n"
        "def __init__(self):\n    self.value = 1",
        "def __getattribute__(self, name):\n    raise RuntimeError()\n\n"
        "def __init__(self):\n    self.value = 1",
        "def __new__(cls):\n    raise RuntimeError()\n\n"
        "def __init__(self):\n    self.value = 1",
    ),
)
def test_unsafe_class_namespace_invalidates_receiver_proof(class_body: str) -> None:
    summary = _summary(
        "class Namespace:\n"
        + textwrap.indent(class_body, "    ")
        + "\n\n    def inspect(self):\n        return self.value\n",
        "Namespace.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == "UNSAFE_CLASS_NAMESPACE"


def test_property_descriptor_invalidates_other_receiver_method_proofs() -> None:
    summary = _summary(
        """
class Descriptor:
    def __init__(self):
        self.value = 1

    @property
    def value(self):
        raise RuntimeError()

    def inspect(self):
        return self.value
""",
        "Descriptor.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category.name == "UNSAFE_CLASS_NAMESPACE"


def test_class_namespace_safety_participates_in_receiver_fingerprint() -> None:
    safe = _summary(
        """
class Storage:
    def __init__(self):
        self.value = 1

    def inspect(self):
        return self.value
""",
        "Storage.inspect",
        module_identity="storage.module",
    )
    unsafe = _summary(
        """
class Storage:
    __slots__ = ()

    def __init__(self):
        self.value = 1

    def inspect(self):
        return self.value
""",
        "Storage.inspect",
        module_identity="storage.module",
    )

    assert safe.receiver.class_fingerprint != unsafe.receiver.class_fingerprint
    assert safe.execution_fingerprint != unsafe.execution_fingerprint


@pytest.mark.parametrize("builtin_name", ("bool", "len"))
def test_wildcard_import_cannot_supply_a_safe_builtin_call(
    builtin_name: str,
) -> None:
    summary = _summary(
        f"""
from shadow_provider import *

class WildcardNamespace:
    def inspect(self, value: int):
        return {builtin_name}(value)
""",
        "WildcardNamespace.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_duplicate_top_level_class_identity_is_never_a_safe_receiver_proof() -> None:
    bundle = analyze_safe_method_summaries(
        ast.parse(
            """
class Shared:
    def inspect(self, value: int):
        return bool(value)

class Shared:
    def inspect(self, value: int):
        return eval(value)
"""
        ),
        module_identity="duplicate.module",
    )

    assert len(bundle.summaries) == 1
    assert all(summary.supported is False for summary in bundle.summaries)
    assert all(
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
        for summary in bundle.summaries
    )
    assert bundle.summary_for("Shared.inspect").supported is False


def test_top_level_class_rebinding_invalidates_receiver_identity() -> None:
    summary = _summary(
        """
class Rebound:
    def inspect(self, value: int):
        return bool(value)

Rebound = replacement
""",
        "Rebound.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "poison_statement",
    (
        "globals()['bool'] = replacement",
        "__builtins__['bool'] = replacement",
    ),
)
def test_dynamic_module_namespace_write_invalidates_builtin_proof(
    poison_statement: str,
) -> None:
    summary = _summary(
        f"""
def replacement(value):
    raise RuntimeError()

{poison_statement}

class Poisoned:
    def inspect(self, value: int):
        return bool(value)
""",
        "Poisoned.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "unsafe_expression",
    (
        "value=explode()",
        "value: explode() = 1",
        "value: int = explode()",
    ),
)
def test_sibling_signature_expression_invalidates_receiver_proof(
    unsafe_expression: str,
) -> None:
    summary = _summary(
        f"""
class SignatureEffect:
    def sibling(self, {unsafe_expression}):
        return value

    def inspect(self):
        return 1
""",
        "SignatureEffect.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_unsafe_module_namespace_changes_receiver_execution_identity() -> None:
    clean = _summary(
        """
class NamespaceIdentity:
    def inspect(self, value: int):
        return bool(value)
""",
        "NamespaceIdentity.inspect",
        module_identity="namespace.identity",
    )
    poisoned = _summary(
        """
globals()['bool'] = replacement

class NamespaceIdentity:
    def inspect(self, value: int):
        return bool(value)
""",
        "NamespaceIdentity.inspect",
        module_identity="namespace.identity",
    )

    assert clean.supported is True
    assert poisoned.supported is False
    assert clean.execution_fingerprint != poisoned.execution_fingerprint


def test_boolean_guard_keeps_every_narrowed_parameter_in_source_order() -> None:
    summary = _summary(
        """
class MultiGuard:
    def inspect(self, alpha: int, beta: int, gamma: int, delta: int):
        if alpha > 0 and beta > 0 and gamma > 0 and delta > 0:
            return 1
        return 0
""",
        "MultiGuard.inspect",
    )

    guard = summary.guards[0]
    assert summary.supported is True
    assert guard.kind == "BOOLEAN_AND"
    assert guard.narrowed_domains == (
        ("parameter:alpha", ("int",)),
        ("parameter:beta", ("int",)),
        ("parameter:gamma", ("int",)),
        ("parameter:delta", ("int",)),
    )


def test_chained_comparison_preserves_all_operators_and_domains() -> None:
    summary = _summary(
        """
class OrderedGuard:
    def inspect(self, lower: int, value: int, upper: int):
        if lower < value < upper:
            return True
        return False
""",
        "OrderedGuard.inspect",
    )

    guard = summary.guards[0]
    assert summary.supported is True
    assert guard.operator == "LESS_THAN|LESS_THAN"
    assert guard.narrowed_domains == (
        ("parameter:lower", ("int",)),
        ("parameter:value", ("int",)),
        ("parameter:upper", ("int",)),
    )


def test_collection_truthiness_guard_carries_receiver_state_proof() -> None:
    summary = _summary(
        """
class CollectionGuard:
    def __init__(self):
        self.entries = []

    def inspect(self):
        if not self.entries:
            return True
        return False
""",
        "CollectionGuard.inspect",
    )

    guard = summary.guards[0]
    assert summary.supported is True
    assert guard.kind == "COLLECTION_TRUTHINESS"
    assert guard.narrowed_domains == (("state:entries", ("EMPTY_LIST",)),)


def test_observer_call_budget_is_enforced_without_truncation() -> None:
    summary = _summary(
        """
class ObserverBudget:
    def __init__(self):
        self.first = []
        self.second = []
        self.third = []

    def inspect(self):
        return len(self.first) + len(self.second) + len(self.third)
""",
        "ObserverBudget.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.observer_call_count == 3
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_mutually_exclusive_observer_calls_use_per_path_budget() -> None:
    summary = _summary(
        """
class BranchObserverBudget:
    def __init__(self):
        self.first = []
        self.second = []
        self.third = []
        self.fourth = []

    def inspect(self, enabled: bool):
        if enabled:
            len(self.first)
            len(self.second)
            return 1
        len(self.third)
        len(self.fourth)
        return 0
""",
        "BranchObserverBudget.inspect",
    )

    assert summary.supported is True
    assert summary.observer_call_count == 2


def test_primitive_bool_calls_share_the_observer_budget() -> None:
    summary = _summary(
        """
class PrimitiveObserverBudget:
    def inspect(self, value: int):
        bool(value)
        bool(value)
        bool(value)
        return value
""",
        "PrimitiveObserverBudget.inspect",
    )

    assert summary.observer_call_count == 3
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_executable_statement_budget_is_enforced_without_truncation() -> None:
    budgets = replace(DEFAULT_SAFE_METHOD_BUDGETS, executable_statements=1)
    summary = _summary(
        """
class StatementBudget:
    def inspect(self, value: int):
        shifted = value + 1
        return shifted
""",
        "StatementBudget.inspect",
        budgets=budgets,
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.statement_count == 2
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


@pytest.mark.parametrize(
    "unsafe_target",
    (
        "external().attribute",
        "mapping()[key()]",
    ),
)
def test_value_less_complex_annotation_target_is_controlled_unsupported(
    unsafe_target: str,
) -> None:
    summary = _summary(
        f"""
class AnnotationTarget:
    def inspect(self):
        {unsafe_target}: int
        return 1
""",
        "AnnotationTarget.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_value_less_local_annotation_remains_a_safe_no_op() -> None:
    summary = _summary(
        """
class LocalAnnotation:
    def inspect(self):
        local: int
        return 1
""",
        "LocalAnnotation.inspect",
    )

    assert summary.supported is True


@pytest.mark.parametrize("attribute_name", ("__class__", "__dict__", "inspect"))
def test_constructor_cannot_assign_protocol_or_method_state(
    attribute_name: str,
) -> None:
    summary = _summary(
        f"""
class ConstructorIdentity:
    def __init__(self):
        self.{attribute_name} = 1

    def inspect(self):
        return 1
""",
        "ConstructorIdentity.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CONSTRUCTOR_STATE
    )


def test_lifecycle_dunder_invalidates_sibling_receiver_proof() -> None:
    summary = _summary(
        """
class Lifecycle:
    def __del__(self):
        open('marker', 'w')

    def inspect(self):
        return 1
""",
        "Lifecycle.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "patch_source",
    (
        "PatchTarget.inspect = replacement",
        "alias = PatchTarget\nalias.inspect = replacement",
        "setattr(PatchTarget, 'inspect', replacement)",
    ),
)
def test_module_level_method_monkeypatch_invalidates_receiver_identity(
    patch_source: str,
) -> None:
    summary = _summary(
        f"""
class PatchTarget:
    def inspect(self, value: int):
        return bool(value)

def replacement(self, value):
    raise RuntimeError()

{patch_source}
""",
        "PatchTarget.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_missing_sibling_annotation_invalidates_class_creation_proof() -> None:
    summary = _summary(
        """
class MissingAnnotation:
    def sibling(self, value: MissingType):
        return value

    def inspect(self):
        return 1
""",
        "MissingAnnotation.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_negative_guard_literal_is_recorded_once() -> None:
    summary = _summary(
        """
class NegativeLiteral:
    def inspect(self, value: int):
        if value < -3:
            return True
        return False
""",
        "NegativeLiteral.inspect",
    )

    assert summary.supported is True
    assert summary.guards[0].literal_values == (("int", -3),)


def test_same_named_parameter_and_state_have_distinct_guard_provenance() -> None:
    summary = _summary(
        """
class DistinctProvenance:
    def __init__(self):
        self.value = 0

    def inspect(self, value: float):
        if self.value < value:
            return True
        return False
""",
        "DistinctProvenance.inspect",
    )

    assert summary.supported is True
    assert summary.guards[0].narrowed_domains == (
        ("state:value", ("int",)),
        ("parameter:value", ("float",)),
    )


def test_nested_observer_argument_is_counted_once() -> None:
    summary = _summary(
        """
class ObserverCount:
    def __init__(self):
        self.left = []
        self.right = []

    def update(self):
        self.left.append(len(self.right))
""",
        "ObserverCount.update",
    )

    assert summary.supported is True
    assert summary.observer_call_count == 1
    assert len(summary.state_effects) == 1


def test_mixed_none_and_primitive_returns_are_not_falsely_narrowed() -> None:
    summary = _summary(
        """
class OptionalReturn:
    def inspect(self, enabled: bool):
        if enabled:
            return None
        return 1
""",
        "OptionalReturn.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


@pytest.mark.parametrize("literal", ("1e309", "-1e309"))
def test_non_finite_float_literal_is_controlled_unsupported(literal: str) -> None:
    summary = _summary(
        f"""
class FiniteOnly:
    def inspect(self):
        hidden = {literal}
        return 1
""",
        "FiniteOnly.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.DOMAIN_CONFLICT


def test_proof_scalar_fields_reject_mutable_values() -> None:
    summary = _summary(
        """
class ScalarProof:
    def inspect(self, value: int):
        if value > 0:
            return 1
        return 0
""",
        "ScalarProof.inspect",
    )

    with pytest.raises(TypeError, match="kind"):
        replace(summary.guards[0], kind=[])
    with pytest.raises(TypeError, match="category"):
        replace(summary, category=[])
    with pytest.raises(TypeError, match="kind"):
        replace(summary.return_summary, kind=[])


def test_guard_proof_rejects_non_boolean_polarities_and_invalid_domains() -> None:
    guard = _summary(
        """
class GuardInvariant:
    def inspect(self, value: int):
        if value > 0:
            return 1
        return 0
""",
        "GuardInvariant.inspect",
    ).guards[0]

    with pytest.raises(TypeError, match="branch_polarities"):
        replace(guard, branch_polarities=(1,))
    with pytest.raises(ValueError, match="narrowed_domains"):
        replace(
            guard,
            narrowed_domains=(("parameter:value", ("object",)),),
        )
    with pytest.raises(TypeError, match="literal_values"):
        replace(guard, literal_values=(("int", "not-an-int"),))


def test_collection_cardinality_budget_limits_ordered_append_effects() -> None:
    budgets = replace(DEFAULT_SAFE_METHOD_BUDGETS, state_effects=4)
    summary = _summary(
        """
class CardinalityBudget:
    def __init__(self):
        self.values = []

    def update(self):
        self.values.append(1)
        self.values.append(2)
        self.values.append(3)
""",
        "CardinalityBudget.update",
        budgets=budgets,
    )

    assert len(summary.state_effects) == 3
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_safe_local_observer_calls_share_the_observer_budget() -> None:
    summary = _summary(
        """
class LocalObserverBudget:
    def read(self):
        return 1

    def inspect(self):
        self.read()
        self.read()
        return self.read()
""",
        "LocalObserverBudget.inspect",
    )

    assert len(summary.local_calls) == 3
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_transitive_local_observer_cost_is_counted_in_the_caller_budget() -> None:
    summary = _summary(
        """
class TransitiveObserverBudget:
    def __init__(self):
        self.first = []
        self.second = []

    def inspect(self):
        return len(self.first) + len(self.second)

    def combined(self):
        len(self.first)
        return self.inspect()
""",
        "TransitiveObserverBudget.combined",
    )

    assert summary.observer_call_count == 3
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_effectful_local_call_also_propagates_its_observer_cost() -> None:
    summary = _summary(
        """
class EffectfulObserverBudget:
    def __init__(self):
        self.first = []
        self.second = []
        self.value = 0

    def helper(self):
        len(self.first)
        len(self.second)
        self.value += 1

    def combined(self):
        len(self.first)
        self.helper()
        return 1
""",
        "EffectfulObserverBudget.combined",
    )

    assert summary.observer_call_count == 3
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_repeated_safe_local_mutator_calls_share_setup_budgets() -> None:
    summary = _summary(
        """
class LocalMutatorBudget:
    def __init__(self):
        self.value = 0

    def update(self):
        self.value += 1

    def inspect(self):
        self.update()
        self.update()
        self.update()
""",
        "LocalMutatorBudget.inspect",
    )

    assert len(summary.local_calls) == 3
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_unbound_module_initializer_invalidates_receiver_proof() -> None:
    summary = _summary(
        """
BROKEN = missing_name

class MissingInitializer:
    def inspect(self):
        return 1
""",
        "MissingInitializer.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_arbitrary_module_call_cannot_poison_a_safe_method_binding() -> None:
    summary = _summary(
        """
def poison():
    global bool
    bool = replacement

poison()

class ModuleCall:
    def inspect(self, value: int):
        return bool(value)
""",
        "ModuleCall.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "initializer",
    ("{[]}", "{[]: 1}", "{([],): 1}"),
)
def test_unhashable_module_container_literal_invalidates_receiver_proof(
    initializer: str,
) -> None:
    summary = _summary(
        f"""
BROKEN = {initializer}

class UnhashableInitializer:
    def inspect(self):
        return 1
""",
        "UnhashableInitializer.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "source",
    (
        """
__name__ = "__main__"

class ReservedBinding:
    def inspect(self):
        return 1

if __name__ == "__main__":
    ReservedBinding.inspect = 7
""",
        """
__builtins__ = {"bool": 1}

class ReservedBinding:
    def inspect(self, enabled: bool):
        return bool(enabled)
""",
    ),
)
def test_reserved_module_runtime_bindings_invalidate_receiver_proof(
    source: str,
) -> None:
    summary = _summary(source, "ReservedBinding.inspect")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "prefix",
    (
        "__annotations__ = None\nmarker: int = 1",
        """
class BrokenAnnotations:
    __annotations__ = None
    marker: int = 1
""",
    ),
)
def test_broken_annotations_namespace_invalidates_later_class_proof(
    prefix: str,
) -> None:
    summary = _summary(
        f"""
{prefix}

class AnnotationConsumer:
    def inspect(self):
        return 1
""",
        "AnnotationConsumer.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_local_class_cannot_shadow_a_primitive_annotation_identity() -> None:
    summary = _summary(
        """
class int:
    pass

class PrimitiveShadow:
    def inspect(self, value: int):
        return value
""",
        "PrimitiveShadow.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    ("shadow_name", "annotation"),
    (("int", "int"), ("list", "list[int]")),
)
def test_prior_class_binding_cannot_shadow_signature_type_identity(
    shadow_name: str,
    annotation: str,
) -> None:
    summary = _summary(
        f"""
class ClassScopeShadow:
    def {shadow_name}(self):
        return 1

    def helper(self, value: {annotation}):
        return 1

    def inspect(self):
        return 1
""",
        "ClassScopeShadow.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_private_method_name_mangling_cannot_hide_state_method_collision() -> None:
    summary = _summary(
        """
class MangledStateCollision:
    def __init__(self):
        self._MangledStateCollision__helper = 0

    def __helper(self):
        return 1

    def inspect(self):
        return self.__helper()
""",
        "MangledStateCollision.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_private_local_name_mangling_cannot_hide_binding_collision() -> None:
    summary = _summary(
        """
class LocalMangleCollision:
    def inspect(self):
        __value = 1
        _LocalMangleCollision__value = 2
        return __value
""",
        "LocalMangleCollision.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize(
    "annotation",
    (
        '"Left" | "Right"',
        'int | "Forward"',
        "int | (str,)",
        "None | None",
        "list[None | None]",
    ),
)
def test_string_forward_reference_cannot_be_a_runtime_union_operand(
    annotation: str,
) -> None:
    summary = _summary(
        f"""
class RuntimeUnion:
    def inspect(self) -> {annotation}:
        return 1
""",
        "RuntimeUnion.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_local_call_return_domain_is_propagated_to_the_caller() -> None:
    summary = _summary(
        """
class LocalReturn:
    def value(self):
        return 1

    def inspect(self):
        return self.value()
""",
        "LocalReturn.inspect",
    )

    assert summary.supported is True
    assert summary.return_summary.kind == "LOCAL_CALL_RESULT"
    assert summary.return_summary.domain is not None
    assert summary.return_summary.domain.admissible_types == ("int",)


def test_local_none_return_cannot_be_merged_as_a_primitive_only_result() -> None:
    summary = _summary(
        """
class LocalOptional:
    def empty(self):
        return None

    def inspect(self, enabled: bool):
        if enabled:
            return self.empty()
        return 1
""",
        "LocalOptional.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


@pytest.mark.parametrize(
    "assignment",
    (
        "self.value = amount + 1",
        "local = amount\n        self.value = local",
    ),
)
def test_direct_state_assignment_rejects_expression_and_local_sources(
    assignment: str,
) -> None:
    summary = _summary(
        f"""
class ClosedAssignment:
    def __init__(self):
        self.value = 0

    def update(self, amount: int):
        {assignment}
""",
        "ClosedAssignment.update",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.UNSAFE_CONSTRUCT


def test_transitive_collection_effects_share_caller_budgets() -> None:
    summary = _summary(
        """
class TransitiveCardinality:
    def __init__(self):
        self.values = []

    def add_pair(self):
        self.values.append(1)
        self.values.append(2)

    def build(self):
        self.add_pair()
        self.add_pair()
""",
        "TransitiveCardinality.build",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_effects_reached_only_through_local_calls_remain_setup_calls() -> None:
    budgets = replace(
        DEFAULT_SAFE_METHOD_BUDGETS,
        local_call_depth=2,
        observer_calls=10,
    )
    summary = _summary(
        """
class TransitiveSetup:
    def __init__(self):
        self.values = []

    def mutate(self):
        self.values.append(1)

    def bridge(self):
        self.mutate()

    def build(self):
        self.bridge()
        self.bridge()
        self.bridge()
""",
        "TransitiveSetup.build",
        budgets=budgets,
    )

    assert summary.observer_call_count == 0
    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert summary.rejection_category is SafeMethodRejectionCategory.BUDGET_EXCEEDED


def test_domain_widening_local_mutator_is_not_promoted_in_the_caller() -> None:
    source = """
class DomainTransition:
    def __init__(self):
        self.total = 0

    def widen(self, amount: float):
        self.total += amount

    def inspect(self, amount: float):
        self.widen(amount)
        return self.total
"""
    callee = _summary(source, "DomainTransition.widen")
    caller = _summary(source, "DomainTransition.inspect")

    assert callee.supported is True
    assert caller.category is SafeMethodCategory.UNSUPPORTED
    assert caller.rejection_category is SafeMethodRejectionCategory.UNSAFE_CALL


def test_non_class_annotation_binding_is_not_a_safe_local_class_identity() -> None:
    summary = _summary(
        """
Alias = 1

class InvalidAlias:
    def helper(self, value: Alias | int):
        return 1

    def inspect(self):
        return 1
""",
        "InvalidAlias.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


@pytest.mark.parametrize("statement", ("break", "continue"))
def test_loop_control_outside_a_loop_is_controlled_unsupported(
    statement: str,
) -> None:
    summary = _summary(
        f"""
class InvalidControl:
    def inspect(self):
        {statement}
""",
        "InvalidControl.inspect",
    )

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNBOUNDED_CONTROL_FLOW
    )


@pytest.mark.parametrize("method_name", ("__init__", "inspect"))
def test_duplicate_signature_names_are_controlled_unsupported(
    method_name: str,
) -> None:
    source = (
        "class DuplicateSignature:\n"
        f"    def {method_name}(self, value: int, value: int):\n"
        "        return value\n"
    )
    if method_name == "__init__":
        source += "\n    def inspect(self):\n        return 1\n"

    summary = _summary(source, "DuplicateSignature.inspect")

    assert summary.category is SafeMethodCategory.UNSUPPORTED
    assert (
        summary.rejection_category
        is SafeMethodRejectionCategory.UNSAFE_CLASS_NAMESPACE
    )


def test_constructor_parameter_to_state_provenance_affects_both_identities() -> None:
    direct = _summary(
        """
class Pair:
    def __init__(self, left: int, right: int):
        self.first = left
        self.second = right

    def inspect(self):
        return self.first
""",
        "Pair.inspect",
        module_identity="pair.module",
    )
    swapped = _summary(
        """
class Pair:
    def __init__(self, left: int, right: int):
        self.first = right
        self.second = left

    def inspect(self):
        return self.first
""",
        "Pair.inspect",
        module_identity="pair.module",
    )

    assert direct.supported is True and swapped.supported is True
    assert direct.execution_fingerprint != swapped.execution_fingerprint
    assert direct.semantic_shape_digest != swapped.semantic_shape_digest
