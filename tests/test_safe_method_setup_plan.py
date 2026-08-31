from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, is_dataclass, replace
from types import ModuleType

import pytest

from analyzer.safe_custom_object import (
    SafeObjectConstructionBlueprint,
    analyze_safe_custom_object_target,
)
from cfg.path_analyzer import ExecutionPath
from analyzer.safe_method_summary import (
    DEFAULT_SAFE_METHOD_BUDGETS,
    SafeMethodBudgets,
    SafeMethodSummary,
    analyze_safe_method_summaries,
)
from generator.safe_method_setup_plan import (
    SafeObjectSetupPlan,
    SafeObjectSlot,
    SafeSetupCall,
    SafeSetupPlanRejection,
    SafeSetupPlanRejectionCategory,
    SafeTargetBinding,
    SafeTypedArgumentBinding,
    analyze_safe_object_setup_context,
    materialize_safe_object_setup_plan,
)
from generator.scenario_generator import Scenario


def _summary(
    source: str,
    qualified_name: str,
    *,
    module_identity: str = "proof.subject",
    budgets: SafeMethodBudgets = DEFAULT_SAFE_METHOD_BUDGETS,
) -> SafeMethodSummary:
    return analyze_safe_method_summaries(
        ast.parse(source),
        module_identity=module_identity,
        budgets=budgets,
    ).summary_for(qualified_name)


def _counter_summary(
    *,
    module_identity: str = "proof.subject",
    class_name: str = "Counter",
    method_name: str = "increase",
    attribute_name: str = "value",
    parameter_name: str = "amount",
) -> SafeMethodSummary:
    return _summary(
        f"""
class {class_name}:
    def __init__(self):
        self.{attribute_name} = 0

    def {method_name}(self, {parameter_name}: int) -> None:
        self.{attribute_name} += {parameter_name}
""",
        f"{class_name}.{method_name}",
        module_identity=module_identity,
    )


def _slot(
    *,
    slot_id: str = "root",
    module_identity: str = "proof.subject",
    class_name: str = "Counter",
    constructor_arguments: tuple[SafeTypedArgumentBinding, ...] = (),
    relation_depth: int = 0,
    owner_slot_id: str | None = None,
    class_fingerprint: str | None = None,
) -> SafeObjectSlot:
    if class_fingerprint is None:
        class_fingerprint = _counter_summary(
            module_identity=module_identity,
            class_name=class_name,
        ).receiver.class_fingerprint
    blueprint = SafeObjectConstructionBlueprint(
        module_identity=module_identity,
        class_name=class_name,
        constructor_arguments=tuple(
            (argument.parameter_name, argument.value)
            for argument in constructor_arguments
        ),
        class_fingerprint=class_fingerprint,
    )
    return SafeObjectSlot(
        slot_id=slot_id,
        blueprint=blueprint,
        relation_depth=relation_depth,
        owner_slot_id=owner_slot_id,
    )


def _primitive(
    name: str,
    type_name: str,
    value: int | float | str | bool,
) -> SafeTypedArgumentBinding:
    return SafeTypedArgumentBinding(
        parameter_name=name,
        type_name=type_name,
        value=value,
    )


def _object_reference(
    name: str,
    slot_id: str,
) -> SafeTypedArgumentBinding:
    return SafeTypedArgumentBinding(
        parameter_name=name,
        type_name="object",
        object_slot_id=slot_id,
    )


def _call(
    summary: SafeMethodSummary,
    *arguments: SafeTypedArgumentBinding,
    receiver_slot_id: str = "root",
    selected_guard_outcomes: tuple[tuple[int, bool], ...] = (),
) -> SafeSetupCall:
    return SafeSetupCall(
        receiver_slot_id=receiver_slot_id,
        method_summary=summary,
        arguments=tuple(arguments),
        selected_guard_outcomes=selected_guard_outcomes,
    )


def _plan(
    *,
    module_identity: str = "proof.subject",
    target_identity: str = "inspect",
    slots: tuple[SafeObjectSlot, ...] | None = None,
    calls: tuple[SafeSetupCall, ...] | None = None,
    bindings: tuple[SafeTargetBinding, ...] | None = None,
    budgets: SafeMethodBudgets = DEFAULT_SAFE_METHOD_BUDGETS,
    normalize_slot_proofs: bool = True,
) -> SafeObjectSetupPlan:
    if slots is None:
        slots = (_slot(module_identity=module_identity),)
    if calls is None:
        calls = (
            _call(
                _counter_summary(module_identity=module_identity),
                _primitive("amount", "int", 1),
            ),
        )
    if bindings is None:
        bindings = (
            SafeTargetBinding(
                parameter_name="subject",
                object_slot_id="root",
            ),
        )
    if normalize_slot_proofs:
        normalized_slots: list[SafeObjectSlot] = []
        for slot in slots:
            receiver_proofs = tuple(
                call.method_summary.receiver
                for call in calls
                if call.receiver_slot_id == slot.slot_id
            )
            if not receiver_proofs:
                receiver_proofs = tuple(
                    local_call.callee_summary.receiver
                    for call in calls
                    for local_call in call.method_summary.local_calls
                    if (
                        local_call.callee_summary.receiver.class_identity
                        == slot.blueprint.class_name
                    )
                )
            if receiver_proofs and len(
                {proof.class_fingerprint for proof in receiver_proofs}
            ) == 1:
                slot = replace(
                    slot,
                    blueprint=replace(
                        slot.blueprint,
                        class_fingerprint=receiver_proofs[0].class_fingerprint,
                    ),
                )
            normalized_slots.append(slot)
        slots = tuple(normalized_slots)
    return SafeObjectSetupPlan(
        module_identity=module_identity,
        target_identity=target_identity,
        object_slots=slots,
        setup_calls=calls,
        target_bindings=bindings,
        budgets=budgets,
    )


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")


def test_setup_plan_models_are_frozen_slotted_and_have_no_public_serializer() -> None:
    plan = _plan()
    instances = (
        plan,
        plan.object_slots[0],
        plan.setup_calls[0],
        plan.setup_calls[0].arguments[0],
        plan.target_bindings[0],
    )

    for instance in instances:
        assert is_dataclass(instance)
        assert instance.__dataclass_params__.frozen is True
        assert not hasattr(instance, "__dict__")
        assert not hasattr(instance, "to_dict")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(instance, "unexpected", True)

    assert isinstance(plan.object_slots, tuple)
    assert isinstance(plan.setup_calls, tuple)
    assert isinstance(plan.target_bindings, tuple)
    assert isinstance(plan.canonical_execution_payload, tuple)
    assert isinstance(plan.canonical_semantic_payload, tuple)


@pytest.mark.parametrize(
    ("module_identity", "class_name", "attribute_name"),
    (
        ("alpha.receiver", "Gauge", "level"),
        ("beta.subject", "Ledger", "balance"),
        ("gamma.container", "Reservoir", "volume"),
    ),
)
def test_setup_plan_materializes_fresh_implicit_receiver_without_public_binding(
    module_identity: str,
    class_name: str,
    attribute_name: str,
) -> None:
    source = f"""
class {class_name}:
    def __init__(self):
        self.{attribute_name} = 0

    def inspect(self) -> int:
        return self.{attribute_name}
"""
    module = ModuleType(module_identity)
    runtime_class = type(
        class_name,
        (),
        {"__init__": lambda self: setattr(self, attribute_name, 0)},
    )
    module.__dict__[class_name] = runtime_class
    receiver_proof = analyze_safe_method_summaries(
        ast.parse(source),
        module_identity=module_identity,
    ).summary_for(f"{class_name}.inspect").receiver
    slot = SafeObjectSlot(
        slot_id="receiver",
        blueprint=SafeObjectConstructionBlueprint(
            module_identity=module_identity,
            class_name=class_name,
            class_fingerprint=receiver_proof.class_fingerprint,
        ),
    )
    plan = SafeObjectSetupPlan(
        module_identity=module_identity,
        target_identity=f"{class_name}.inspect",
        object_slots=(slot,),
        setup_calls=(),
        target_bindings=(),
        receiver_slot_id="receiver",
    )

    first_arguments, first_receiver = materialize_safe_object_setup_plan(
        plan,
        module,
        include_receiver=True,
    )
    second_arguments, second_receiver = materialize_safe_object_setup_plan(
        plan,
        module,
        include_receiver=True,
    )

    assert first_arguments == second_arguments == {}
    assert type(first_receiver) is runtime_class
    assert type(second_receiver) is runtime_class
    assert first_receiver is not second_receiver
    assert "self" not in plan.canonical_execution_payload
    assert not hasattr(plan, "to_dict")


def test_setup_plan_receiver_binding_changes_canonical_fingerprints() -> None:
    parameter_plan = _plan()
    receiver_plan = replace(
        parameter_plan,
        target_bindings=(),
        receiver_slot_id="root",
    )
    repeated_receiver_plan = replace(
        parameter_plan,
        target_bindings=(),
        receiver_slot_id="root",
    )

    assert receiver_plan.execution_fingerprint == repeated_receiver_plan.execution_fingerprint
    assert receiver_plan.semantic_shape_digest == repeated_receiver_plan.semantic_shape_digest
    assert receiver_plan.execution_fingerprint != parameter_plan.execution_fingerprint


def test_setup_plan_rejects_unknown_implicit_receiver_slot() -> None:
    with pytest.raises(SafeSetupPlanRejection):
        replace(_plan(), target_bindings=(), receiver_slot_id="missing")


def test_equivalent_plan_has_stable_canonical_identities_and_private_repr() -> None:
    constructor = (_primitive("value", "int", 41),)
    first = _plan(
        slots=(
            _slot(constructor_arguments=constructor),
        ),
    )
    second = _plan(
        slots=(
            _slot(constructor_arguments=constructor),
        ),
    )

    assert first == second
    assert first.execution_fingerprint == second.execution_fingerprint
    assert first.semantic_shape_digest == second.semantic_shape_digest
    assert first.pre_state_fingerprint == second.pre_state_fingerprint
    assert first.post_state_fingerprint == second.post_state_fingerprint
    for value in (
        first.execution_fingerprint,
        first.semantic_shape_digest,
        first.pre_state_fingerprint,
        first.post_state_fingerprint,
    ):
        _assert_sha256(value)

    rendered = repr(first)
    assert "41" not in rendered
    assert "constructor_arguments" not in rendered
    assert "object at 0x" not in rendered


def test_setup_order_and_typed_values_participate_in_execution_identity() -> None:
    source = """
class Register:
    def __init__(self):
        self.value = 0

    def increase(self, amount: int) -> None:
        self.value += amount

    def decrease(self, amount: int) -> None:
        self.value -= amount
"""
    bundle = analyze_safe_method_summaries(
        ast.parse(source),
        module_identity="proof.subject",
    )
    increase = bundle.summary_for("Register.increase")
    decrease = bundle.summary_for("Register.decrease")
    slot = _slot(class_name="Register")
    first_call = _call(
        increase,
        _primitive("amount", "int", 3),
    )
    second_call = _call(
        decrease,
        _primitive("amount", "int", 1),
    )
    forward = _plan(
        slots=(slot,),
        calls=(first_call, second_call),
    )
    reversed_order = _plan(
        slots=(slot,),
        calls=(second_call, first_call),
    )
    other_value = _plan(
        slots=(slot,),
        calls=(
            replace(
                first_call,
                arguments=(_primitive("amount", "int", 4),),
            ),
            second_call,
        ),
    )

    assert forward.execution_fingerprint != reversed_order.execution_fingerprint
    assert forward.post_state_fingerprint != reversed_order.post_state_fingerprint
    assert forward.semantic_shape_digest != reversed_order.semantic_shape_digest
    assert forward.execution_fingerprint != other_value.execution_fingerprint
    assert forward.semantic_shape_digest != other_value.semantic_shape_digest


def test_module_identity_separates_execution_without_changing_semantic_shape() -> None:
    first = _plan(module_identity="first_package.subject")
    second = _plan(module_identity="second_package.subject")

    assert first.execution_fingerprint != second.execution_fingerprint
    assert first.pre_state_fingerprint != second.pre_state_fingerprint
    assert first.post_state_fingerprint != second.post_state_fingerprint
    assert first.semantic_shape_digest == second.semantic_shape_digest
    assert "first_package.subject" not in repr(first)


@pytest.mark.parametrize(
    (
        "module_identity",
        "class_name",
        "method_name",
        "attribute_name",
        "parameter_name",
        "target_name",
        "target_parameter",
    ),
    (
        (
            "rename.alpha",
            "Accumulator",
            "advance",
            "level",
            "delta",
            "inspect",
            "subject",
        ),
        (
            "rename.beta",
            "Gauge",
            "shift",
            "reading",
            "step",
            "evaluate",
            "candidate",
        ),
        (
            "rename.gamma",
            "Ledger",
            "record",
            "balance",
            "change",
            "review",
            "payload",
        ),
    ),
)
def test_three_renamed_plans_share_semantic_shape(
    module_identity: str,
    class_name: str,
    method_name: str,
    attribute_name: str,
    parameter_name: str,
    target_name: str,
    target_parameter: str,
    request: pytest.FixtureRequest,
) -> None:
    summary = _counter_summary(
        module_identity=module_identity,
        class_name=class_name,
        method_name=method_name,
        attribute_name=attribute_name,
        parameter_name=parameter_name,
    )
    plan = _plan(
        module_identity=module_identity,
        target_identity=target_name,
        slots=(
            _slot(
                module_identity=module_identity,
                class_name=class_name,
            ),
        ),
        calls=(
            _call(
                summary,
                _primitive(parameter_name, "int", 3),
            ),
        ),
        bindings=(
            SafeTargetBinding(
                parameter_name=target_parameter,
                object_slot_id="root",
            ),
        ),
    )

    request.node.user_properties.append(
        ("execution_fingerprint", plan.execution_fingerprint)
    )
    request.node.user_properties.append(
        ("semantic_shape_digest", plan.semantic_shape_digest)
    )
    assert plan.semantic_shape_digest


def test_three_renamed_plan_identities_are_equal_only_semantically() -> None:
    variants = (
        ("rename.alpha", "Accumulator", "advance", "level", "delta", "inspect", "subject"),
        ("rename.beta", "Gauge", "shift", "reading", "step", "evaluate", "candidate"),
        ("rename.gamma", "Ledger", "record", "balance", "change", "review", "payload"),
    )
    plans: list[SafeObjectSetupPlan] = []
    for module, class_name, method, attribute, parameter, target, target_parameter in variants:
        summary = _counter_summary(
            module_identity=module,
            class_name=class_name,
            method_name=method,
            attribute_name=attribute,
            parameter_name=parameter,
        )
        plans.append(
            _plan(
                module_identity=module,
                target_identity=target,
                slots=(
                    _slot(module_identity=module, class_name=class_name),
                ),
                calls=(
                    _call(summary, _primitive(parameter, "int", 3)),
                ),
                bindings=(
                    SafeTargetBinding(
                        parameter_name=target_parameter,
                        object_slot_id="root",
                    ),
                ),
            )
        )

    assert len({plan.semantic_shape_digest for plan in plans}) == 1
    assert len({plan.execution_fingerprint for plan in plans}) == 3


@pytest.mark.parametrize(
    ("type_name", "value"),
    (
        ("int", 7),
        ("float", 7.5),
        ("str", "ready"),
        ("bool", True),
    ),
)
def test_primitive_binding_uses_exact_runtime_types(
    type_name: str,
    value: int | float | str | bool,
) -> None:
    binding = _primitive("value", type_name, value)

    assert binding.type_name == type_name
    assert type(binding.value).__name__ == type_name


def test_bool_is_not_accepted_as_an_int_and_unresolved_domains_are_rejected() -> None:
    with pytest.raises(
        SafeSetupPlanRejection,
    ) as bool_error:
        _primitive("value", "int", True)
    assert (
        bool_error.value.category
        is SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED
    )

    with pytest.raises(SafeSetupPlanRejection) as ambiguous_error:
        SafeTypedArgumentBinding(
            parameter_name="value",
            type_name="int|float",
            value=1,
        )
    assert (
        ambiguous_error.value.category
        is SafeSetupPlanRejectionCategory.PRIMITIVE_DOMAIN_UNRESOLVED
    )


def test_nested_object_binding_is_owned_local_and_materialized_fresh() -> None:
    source = """
class Element:
    def __init__(self, code: str, stock: int):
        self.code = code
        self.stock = stock

    def decrease(self, amount: int) -> None:
        self.stock -= amount

class Container:
    def __init__(self):
        self.entries = {}

    def attach(self, quantity: int) -> None:
        self.entries["fixed"] = quantity
"""
    bundle = analyze_safe_method_summaries(
        ast.parse(source),
        module_identity="proof.nested",
    )
    attach = bundle.summary_for("Container.attach")
    root = _slot(
        module_identity="proof.nested",
        class_name="Container",
    )
    nested = _slot(
        slot_id="nested_001",
        module_identity="proof.nested",
        class_name="Element",
        constructor_arguments=(
            _primitive("code", "str", "x"),
            _primitive("stock", "int", 4),
        ),
        relation_depth=1,
        owner_slot_id="root",
    )
    plan = _plan(
        module_identity="proof.nested",
        slots=(root, nested),
        calls=(
            _call(
                attach,
                _primitive("quantity", "int", 1),
            ),
        ),
        bindings=(
            SafeTargetBinding(
                parameter_name="subject",
                object_slot_id="root",
            ),
            SafeTargetBinding(
                parameter_name="nested",
                object_slot_id="nested_001",
            ),
        ),
    )

    class Element:
        def __init__(self, code: str, stock: int) -> None:
            self.code = code
            self.stock = stock

        def decrease(self, amount: int) -> None:
            self.stock -= amount

    class Container:
        def __init__(self) -> None:
            self.entries: dict[str, int] = {}

        def attach(self, quantity: int) -> None:
            self.entries["fixed"] = quantity

    module = ModuleType("proof.nested")
    module.Element = Element
    module.Container = Container
    materialized = materialize_safe_object_setup_plan(plan, module)
    container = materialized["subject"]

    assert isinstance(container, Container)
    assert isinstance(materialized["nested"], Element)
    assert container.entries == {"fixed": 1}


@pytest.mark.parametrize(
    ("slots", "expected_category"),
    (
        (
            (
                _slot(),
                _slot(
                    slot_id="foreign",
                    module_identity="other.subject",
                    class_name="Other",
                    relation_depth=1,
                    owner_slot_id="root",
                ),
            ),
            SafeSetupPlanRejectionCategory.CROSS_MODULE_OBJECT,
        ),
        (
            (
                _slot(),
                _slot(
                    slot_id="shared",
                    class_name="Shared",
                    relation_depth=1,
                    owner_slot_id="missing",
                ),
            ),
            SafeSetupPlanRejectionCategory.UNSAFE_OBJECT_GRAPH,
        ),
        (
            (
                _slot(),
                _slot(
                    slot_id="depth_001",
                    class_name="DepthOne",
                    relation_depth=1,
                    owner_slot_id="root",
                ),
                _slot(
                    slot_id="depth_002",
                    class_name="DepthTwo",
                    relation_depth=2,
                    owner_slot_id="depth_001",
                ),
                _slot(
                    slot_id="too_deep",
                    class_name="DepthThree",
                    relation_depth=3,
                    owner_slot_id="depth_002",
                ),
            ),
            SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED,
        ),
    ),
)
def test_unsafe_nested_object_graphs_are_controlled_rejections(
    slots: tuple[SafeObjectSlot, ...],
    expected_category: SafeSetupPlanRejectionCategory,
) -> None:
    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(slots=slots, calls=())

    assert error.value.category is expected_category


def test_top_level_setup_and_repeated_mutator_budgets_are_not_truncated() -> None:
    summary = _counter_summary()
    calls = tuple(
        _call(summary, _primitive("amount", "int", value))
        for value in (1, 2, 3)
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(calls=calls)

    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED
    assert "3" in str(error.value)


def test_object_count_budget_is_not_silently_truncated() -> None:
    slots = tuple(
        _slot(
            slot_id=f"slot_{index}",
            class_name=f"Object{index}",
        )
        for index in range(DEFAULT_SAFE_METHOD_BUDGETS.max_objects + 1)
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=slots,
            calls=(),
            bindings=(
                SafeTargetBinding(
                    parameter_name="subject",
                    object_slot_id="slot_0",
                ),
            ),
        )

    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED


def test_collection_cardinality_budget_counts_transitive_collection_effects() -> None:
    append = _summary(
        """
class Queue:
    def __init__(self):
        self.values = []

    def append(self, value: int) -> None:
        self.values.append(value)
""",
        "Queue.append",
    )
    calls = tuple(
        _call(append, _primitive("value", "int", value))
        for value in (1, 2, 3)
    )
    permissive_calls = replace(
        DEFAULT_SAFE_METHOD_BUDGETS,
        top_level_setup_calls=4,
        repeated_mutator_calls=4,
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Queue"),),
            calls=calls,
            budgets=permissive_calls,
        )

    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED


def test_two_setup_calls_do_not_hide_transitive_state_effect_budget() -> None:
    summary = _summary(
        """
class Pair:
    def __init__(self):
        self.left = 0
        self.right = 0

    def mutate(self, amount: int) -> None:
        self.left += amount
        self.right += amount

    def apply(self, amount: int) -> None:
        self.mutate(amount)
""",
        "Pair.apply",
    )
    assert summary.supported is True
    assert len(summary.local_calls) == 1
    assert summary.state_effects == ()

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Pair"),),
            calls=(
                _call(summary, _primitive("amount", "int", 1)),
                _call(summary, _primitive("amount", "int", 2)),
            ),
        )

    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED


def test_unsupported_or_guard_incompatible_method_is_a_controlled_rejection() -> None:
    unsupported = _summary(
        """
class Dynamic:
    def __init__(self):
        self.value = 0

    def run(self, callback):
        return callback(self.value)
""",
        "Dynamic.run",
    )

    with pytest.raises(SafeSetupPlanRejection) as unsupported_error:
        _plan(
            slots=(_slot(class_name="Dynamic"),),
            calls=(_call(unsupported),),
        )
    assert (
        unsupported_error.value.category
        is SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF
    )

    guarded = _summary(
        """
class Guarded:
    def __init__(self):
        self.value = 0

    def apply(self, amount: int) -> None:
        if amount > 0:
            self.value += amount
""",
        "Guarded.apply",
    )
    with pytest.raises(SafeSetupPlanRejection) as guard_error:
        _plan(
            slots=(_slot(class_name="Guarded"),),
            calls=(
                _call(
                    guarded,
                    _primitive("amount", "int", 1),
                    selected_guard_outcomes=((99, True),),
                ),
            ),
        )
    assert (
        guard_error.value.category
        is SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION
    )


def test_selected_guard_outcomes_must_cover_effect_guard_context() -> None:
    guarded = _summary(
        """
class Guarded:
    def __init__(self):
        self.value = 0

    def apply(self, amount: int) -> None:
        if amount > 0:
            self.value += amount
""",
        "Guarded.apply",
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Guarded"),),
            calls=(
                _call(
                    guarded,
                    _primitive("amount", "int", 1),
                    selected_guard_outcomes=(),
                ),
            ),
        )

    assert (
        error.value.category
        is SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION
    )


def test_materialization_creates_a_fresh_instance_and_replays_real_setup_each_time() -> None:
    class RuntimeCounter:
        def __init__(self, value: int) -> None:
            self.value = value

        def increase(self, amount: int) -> None:
            self.value += amount

    module = ModuleType("runtime.subject")
    module.RuntimeCounter = RuntimeCounter
    summary = _counter_summary(
        module_identity="runtime.subject",
        class_name="RuntimeCounter",
    )
    plan = _plan(
        module_identity="runtime.subject",
        slots=(
            _slot(
                module_identity="runtime.subject",
                class_name="RuntimeCounter",
                constructor_arguments=(_primitive("value", "int", 0),),
            ),
        ),
        calls=(
            _call(summary, _primitive("amount", "int", 1)),
        ),
    )

    first = materialize_safe_object_setup_plan(plan, module)["subject"]
    second = materialize_safe_object_setup_plan(plan, module)["subject"]

    assert isinstance(first, RuntimeCounter)
    assert isinstance(second, RuntimeCounter)
    assert first is not second
    assert first.value == second.value == 1
    first.increase(8)
    assert first.value == 9
    assert second.value == 1


def test_materialization_does_not_use_state_injection() -> None:
    class RuntimeCounter:
        def __init__(self) -> None:
            self.value = 0

        def increase(self, amount: int) -> None:
            self.value += amount

    module = ModuleType("runtime.safe_subject")
    module.RuntimeCounter = RuntimeCounter
    summary = _counter_summary(
        module_identity="runtime.safe_subject",
        class_name="RuntimeCounter",
    )
    plan = _plan(
        module_identity="runtime.safe_subject",
        slots=(
            _slot(
                module_identity="runtime.safe_subject",
                class_name="RuntimeCounter",
            ),
        ),
        calls=(
            _call(summary, _primitive("amount", "int", 2)),
        ),
    )

    materialized = materialize_safe_object_setup_plan(plan, module)
    implementation = inspect.getsource(materialize_safe_object_setup_plan)

    assert materialized["subject"].value == 2
    assert "setattr(" not in implementation
    assert "receiver.__dict__" not in implementation
    assert "instance.__dict__" not in implementation


def _bounded_runtime_source(
    *,
    item_class: str = "Element",
    owner_class: str = "Container",
    target_name: str = "evaluate",
) -> str:
    return f'''
class {item_class}:
    def __init__(self, code: str, value: int, stock: int):
        self.code = code
        self.value = value
        self.stock = stock

    def adjust(self, amount: int):
        if self.stock + amount < 0:
            raise ValueError("invalid")
        self.stock += amount

class {owner_class}:
    def __init__(self):
        self.entries = {{}}

    def attach(self, item, quantity: int):
        if quantity <= 0:
            raise ValueError("invalid")
        if item.stock < quantity:
            return False
        if item.code in self.entries:
            self.entries[item.code]["quantity"] += quantity
        else:
            self.entries[item.code] = {{"item": item, "quantity": quantity}}
        item.adjust(-quantity)
        return True

    def measure(self, mode: str = ""):
        total = 0
        for record in self.entries.values():
            item = record["item"]
            quantity = record["quantity"]
            total += item.value * quantity
        return round(total, 2)

    def label(self):
        if not self.entries:
            return None
        selected = None
        for record in self.entries.values():
            item = record["item"]
            if selected is None:
                selected = item
            elif item.value > selected.value:
                selected = item
        return selected.code

def {target_name}(subject, mode):
    if not subject.entries:
        return 0
    total = subject.measure(mode)
    if total >= 10:
        return subject.label()
    return "low"
'''


def test_context_rewrites_only_proven_path_markers_and_builds_real_setup() -> None:
    tree = ast.parse(_bounded_runtime_source())
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="examples.runtime_slice",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context_reason is None and context is not None
    marker = "__rlut_unsupported_custom_object_method_call_7f4d3a"
    path = ExecutionPath(
        node_ids=[1, 2, 3, 4, 5],
        edge_labels=["False", None, "True", None],
        node_labels=[
            f"not {marker}('STATE_TRUTHINESS', 'subject', 'entries')",
            f"total = {marker}('METHOD', 'subject', 'measure', mode)",
            "total >= 10",
            f"return {marker}('METHOD', 'subject', 'label')",
            "END",
        ],
        node_types=["if", "Assign", "if", "return", "end"],
        line_numbers=[1, 2, 3, 4, None],
    )
    rewritten = context.rewrite_path(path)
    assert all(marker not in label for label in rewritten.node_labels)

    names, types = context.analysis_metadata(("mode",), {"mode": "str"})
    assert context.state_analysis_name in names
    assert types[context.state_analysis_name] == "bool"
    values = {
        context.state_analysis_name: True,
        context.method_analysis_names["measure"]: 10,
        context.method_analysis_names["label"]: "unit",
        "mode": "",
    }
    scenario = Scenario(
        scenario_id="runtime_slice_001",
        name="runtime slice",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1,),
        edge_labels=(),
        contains_loop=False,
        contains_exception=False,
        description="runtime slice",
        keyword_arguments=tuple(values.items()),
        expected_result="unit",
    )
    bound = context.bind_scenario(scenario)

    assert bound.setup_plan is not None
    assert len(bound.setup_plan.object_slots) == 2
    assert len(bound.setup_plan.setup_calls) == 1
    assert context.state_analysis_name not in dict(bound.keyword_arguments)


@pytest.mark.parametrize(
    ("item_class", "owner_class", "target_name", "module_identity"),
    (
        ("Element", "Container", "evaluate", "fixtures.first"),
        ("Parcel", "Warehouse", "inspect", "fixtures.second"),
        ("Token", "Registry", "classify", "fixtures.third"),
    ),
)
def test_context_is_general_across_renamed_equivalent_sources(
    item_class: str,
    owner_class: str,
    target_name: str,
    module_identity: str,
) -> None:
    tree = ast.parse(
        _bounded_runtime_source(
            item_class=item_class,
            owner_class=owner_class,
            target_name=target_name,
        )
    )
    spec, reason = analyze_safe_custom_object_target(tree, target_name)
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity=module_identity,
        target_name=target_name,
        custom_object_spec=spec,
    )

    assert context_reason is None and context is not None
    assert context.semantic_shape_digest


def test_context_rejects_constructor_attribute_parameter_name_mismatch() -> None:
    source = _bounded_runtime_source().replace(
        "def __init__(self, code: str, value: int, stock: int):\n"
        "        self.code = code",
        "def __init__(self, identifier: str, value: int, stock: int):\n"
        "        self.code = identifier",
        1,
    )
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.constructor_alias",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason == "UNPROVEN_BOUNDED_SETUP"


def test_context_does_not_infer_method_domain_from_unrelated_receiver_call() -> None:
    source = _bounded_runtime_source().replace(
        'def measure(self, mode: str = ""):',
        "def measure(self, mode):",
        1,
    )
    source = (
        "class Decoy:\n"
        "    def measure(self, value):\n"
        "        return value\n\n"
        "def unrelated(receiver):\n"
        "    return receiver.measure(17)\n\n"
        + source
    )
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.receiver_provenance",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason == "UNKNOWN_PRIMITIVE_DOMAIN"


def test_context_infers_untyped_setup_domain_from_proven_local_receiver_call() -> None:
    source = _bounded_runtime_source().replace(
        "def attach(self, item, quantity: int):",
        "def attach(self, item, quantity):",
        1,
    )
    source += (
        "\ndef prove_receiver_domain():\n"
        "    entry = Element('entry', 19, 3)\n"
        "    owner = Container()\n"
        "    owner.attach(entry, 1)\n"
    )
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.proven_receiver",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context_reason is None
    assert context is not None
    assert context.setup_quantity_type == "int"


@pytest.mark.parametrize(
    "source",
    (
        _bounded_runtime_source().replace(
            "        item.adjust(-quantity)\n",
            "        external_effect()\n"
            "        item.adjust(-quantity)\n",
            1,
        ),
        _bounded_runtime_source().replace(
            "        self.stock = stock\n",
            "        self.stock = stock\n"
            "        external_effect()\n",
            1,
        ),
        _bounded_runtime_source().replace(
            "        self.stock += amount\n",
            "        self.stock += amount\n"
            "        external_state.value = amount\n",
            1,
        ),
        _bounded_runtime_source().replace(
            "        return round(total, 2)\n",
            "        self.audit_value = total\n"
            "        return round(total, 2)\n",
            1,
        ),
    ),
)
def test_context_rejects_unproven_setup_or_observer_side_effects(
    source: str,
) -> None:
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.side_effect_boundary",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason in {
        "UNPROVEN_BOUNDED_SETUP",
        "UNPROVEN_OBSERVER_CALL",
    }


def test_context_rejects_nested_class_inheritance() -> None:
    source = _bounded_runtime_source().replace(
        "class Element:\n",
        "class Base:\n"
        "    pass\n\n"
        "class Element(Base):\n",
        1,
    )
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.inherited_nested",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason == "UNPROVEN_BOUNDED_SETUP"


def test_context_rejects_nested_dynamic_attribute_protocol() -> None:
    source = _bounded_runtime_source().replace(
        "    def adjust(self, amount: int):\n",
        "    def __setattr__(self, name, value):\n"
        "        external_effect()\n\n"
        "    def adjust(self, amount: int):\n",
        1,
    )
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.dynamic_protocol",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason == "UNPROVEN_BOUNDED_SETUP"


@pytest.mark.parametrize(
    "source",
    (
        _bounded_runtime_source().replace(
            "if quantity <= 0:",
            "if quantity >= 0:",
            1,
        ),
        _bounded_runtime_source().replace(
            "if item.code in self.entries:",
            "if item.code not in self.entries:",
            1,
        ),
    ),
)
def test_context_rejects_setup_shapes_with_unproven_guard_polarity(
    source: str,
) -> None:
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.guard_polarity",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason == "UNPROVEN_BOUNDED_SETUP"


def test_context_rejects_duplicate_observer_method_identity() -> None:
    source = _bounded_runtime_source().replace(
        "return subject.label()",
        "return subject.measure(mode)",
        1,
    )
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.duplicate_observer",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason == "UNPROVEN_OBSERVER_CALL"


@pytest.mark.parametrize(
    "unsafe_source",
    (
        _bounded_runtime_source().replace(
            "    def attach(self, item, quantity: int):",
            "    def __getattribute__(self, name):\n"
            "        return object.__getattribute__(self, name)\n\n"
            "    def attach(self, item, quantity: int):",
            1,
        ),
        _bounded_runtime_source().replace(
            "    def attach(self, item, quantity: int):",
            "    @audit\n    def attach(self, item, quantity: int):",
            1,
        ),
        _bounded_runtime_source().replace(
            "    def measure(self, mode: str = \"\"):",
            "    def attach(self, item, quantity: int):\n"
            "        return True\n\n"
            "    def measure(self, mode: str = \"\"):",
            1,
        ),
    ),
)
def test_unsafe_d1_class_or_constructor_proof_is_never_promoted_to_setup(
    unsafe_source: str,
) -> None:
    tree = ast.parse(unsafe_source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    if spec is None:
        assert reason is not None
        return
    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.unsafe_promotion",
        target_name="evaluate",
        custom_object_spec=spec,
    )
    assert context is None
    assert context_reason is not None


@pytest.mark.parametrize(
    ("budget_transform", "expected_fragment"),
    (
        (
            lambda budgets: replace(budgets, guards=1),
            "guard outcome",
        ),
        (
            lambda budgets: replace(budgets, executable_statements=1),
            "executable statement",
        ),
    ),
)
def test_aggregate_setup_footprints_are_rejected_without_truncation(
    budget_transform,
    expected_fragment: str,
) -> None:
    base = _summary(
        """
class Aggregate:
    def __init__(self):
        self.value = 0

    def nested(self, amount: int) -> None:
        self.value += amount

    def apply(self, amount: int) -> None:
        if amount > 0:
            self.nested(amount)
        if amount < 10:
            self.value += amount
""",
        "Aggregate.apply",
    )
    assert len(base.local_calls) == 1
    budgets = budget_transform(DEFAULT_SAFE_METHOD_BUDGETS)
    selected_guards = tuple((index, True) for index in range(len(base.guards)))
    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Aggregate"),),
            calls=(
                _call(
                    base,
                    _primitive("amount", "int", 1),
                    selected_guard_outcomes=selected_guards,
                ),
            ),
            budgets=budgets,
        )
    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED
    assert expected_fragment in str(error.value)


def test_local_call_depth_budget_uses_analyzer_derived_transitive_footprint() -> None:
    analysis_budgets = replace(
        DEFAULT_SAFE_METHOD_BUDGETS,
        local_call_depth=2,
    )
    summary = _summary(
        """
class Chain:
    def __init__(self):
        self.value = 0

    def nested(self, amount: int) -> None:
        self.value += amount

    def middle(self, amount: int) -> None:
        self.nested(amount)

    def apply(self, amount: int) -> None:
        self.middle(amount)
""",
        "Chain.apply",
        budgets=analysis_budgets,
    )
    assert summary.supported is True
    assert summary.budget_footprint.local_call_depth == 2

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Chain"),),
            calls=(
                _call(
                    summary,
                    _primitive("amount", "int", 1),
                ),
            ),
        )

    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED
    assert "local call depth" in str(error.value)


def test_observer_budget_uses_analyzer_derived_transitive_footprint() -> None:
    summary = _summary(
        """
class Reader:
    def read(self) -> int:
        return 1

    def inspect(self) -> int:
        self.read()
        return self.read()
""",
        "Reader.inspect",
    )
    assert summary.supported is True
    assert summary.budget_footprint.observer_calls == 2

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Reader"),),
            calls=(_call(summary),),
            budgets=replace(DEFAULT_SAFE_METHOD_BUDGETS, observer_calls=1),
        )

    assert error.value.category is SafeSetupPlanRejectionCategory.BUDGET_EXCEEDED
    assert "observer call" in str(error.value)


def test_path_scoped_empty_and_nonempty_bindings_do_not_share_setup_state() -> None:
    tree = ast.parse(_bounded_runtime_source())
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None
    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.path_isolation",
        target_name="evaluate",
        custom_object_spec=spec,
    )
    assert context_reason is None and context is not None

    def bind(path_index: int, populated: bool) -> Scenario:
        values = {
            context.state_analysis_name: populated,
            "mode": "",
            **{
                observer.analysis_name: (
                    "unit" if observer.return_type == "str" else 10
                )
                for observer in context.observers
            },
        }
        return context.bind_scenario(
            Scenario(
                scenario_id=f"path_{path_index}",
                name=f"path {path_index}",
                path_index=path_index,
                priority_rank=path_index,
                priority_level="High",
                dqm_score=100.0,
                node_ids=(path_index,),
                edge_labels=(),
                contains_loop=False,
                contains_exception=False,
                description="path isolation",
                keyword_arguments=tuple(values.items()),
                expected_result="unit" if populated else 0,
            )
        )

    empty = bind(1, False)
    populated = bind(2, True)
    empty_again = bind(3, False)
    assert empty.setup_plan is not None
    assert populated.setup_plan is not None
    assert empty_again.setup_plan is not None
    assert empty.setup_plan.setup_calls == ()
    assert len(populated.setup_plan.setup_calls) == 1
    assert empty_again.setup_plan.setup_calls == ()
    assert (
        empty.setup_plan.execution_fingerprint
        == empty_again.setup_plan.execution_fingerprint
    )
    assert (
        empty.setup_plan.execution_fingerprint
        != populated.setup_plan.execution_fingerprint
    )
    assert context.state_analysis_name not in dict(empty.keyword_arguments)
    assert context.state_analysis_name not in dict(populated.keyword_arguments)


def test_guard_outcome_must_match_bound_primitive_witness() -> None:
    summary = _summary(
        """
class Guarded:
    def __init__(self):
        self.value = 0

    def apply(self, amount: int) -> None:
        if amount > 0:
            self.value += amount
""",
        "Guarded.apply",
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(_slot(class_name="Guarded"),),
            calls=(
                _call(
                    summary,
                    _primitive("amount", "int", -1),
                    selected_guard_outcomes=((0, True),),
                ),
            ),
        )

    assert (
        error.value.category
        is SafeSetupPlanRejectionCategory.INVALID_GUARD_SELECTION
    )


@pytest.mark.parametrize(
    "source",
    (
        _bounded_runtime_source().replace(
            "def attach(self, item, quantity: int):",
            "def attach(self, item, quantity: int, extra):",
            1,
        ),
        _bounded_runtime_source().replace(
            "def adjust(self, amount: int):",
            "def adjust(self, amount: int, extra):",
            1,
        ),
        _bounded_runtime_source().replace(
            'self.entries[item.code] = {"item": item, "quantity": quantity}',
            'self.entries[item.code] = {"item": item, "quantity": quantity, '
            '"extra": item.dynamic}',
            1,
        ),
    ),
)
def test_setup_signature_and_owned_record_payload_are_exactly_bounded(
    source: str,
) -> None:
    tree = ast.parse(source)
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None

    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.exact_setup_boundary",
        target_name="evaluate",
        custom_object_spec=spec,
    )

    assert context is None
    assert context_reason in {
        "UNPROVEN_BOUNDED_SETUP",
        "UNKNOWN_PRIMITIVE_DOMAIN",
    }


def test_setup_call_rejects_receiver_without_constructor_safety_proof() -> None:
    summary = _counter_summary()
    forged = replace(
        summary,
        receiver=replace(summary.receiver, constructor_safe=False),
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _call(forged, _primitive("amount", "int", 1))

    assert (
        error.value.category
        is SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF
    )


def test_setup_plan_binds_receiver_proof_to_object_slot_fingerprint() -> None:
    summary = _counter_summary()
    blueprint = SafeObjectConstructionBlueprint(
        module_identity="proof.subject",
        class_name="Counter",
        class_fingerprint=summary.receiver.class_fingerprint,
    )
    slot = SafeObjectSlot(slot_id="root", blueprint=blueprint)
    forged = replace(
        summary,
        receiver=replace(summary.receiver, class_fingerprint="f" * 64),
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        _plan(
            slots=(slot,),
            calls=(
                _call(
                    forged,
                    _primitive("amount", "int", 1),
                ),
            ),
            normalize_slot_proofs=False,
        )

    assert (
        error.value.category
        is SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF
    )


def test_transitive_budget_footprint_cannot_be_forged() -> None:
    summary = _summary(
        """
class Aggregate:
    def __init__(self):
        self.value = 0

    def nested(self, amount: int) -> None:
        self.value += amount

    def apply(self, amount: int) -> None:
        self.nested(amount)
""",
        "Aggregate.apply",
    )
    assert summary.state_effects == ()
    assert summary.budget_footprint.state_effects == 1

    forged_footprint = replace(
        summary.budget_footprint,
        state_effects=0,
        state_domain_types=(),
        collection_effects=(),
        mutator_calls=(),
        executable_statements=summary.statement_count,
        closure_complete=True,
    )

    with pytest.raises(ValueError, match="budget_footprint"):
        replace(summary, budget_footprint=forged_footprint)


def test_local_call_depth_is_derived_from_callee_summary() -> None:
    summary = _summary(
        """
class Chain:
    def nested(self) -> int:
        return 1

    def apply(self) -> int:
        return self.nested()
""",
        "Chain.apply",
    )
    local_call = summary.local_calls[0]

    with pytest.raises(ValueError, match="call_depth"):
        replace(local_call, call_depth=local_call.call_depth + 1)


def test_local_call_rejects_unsupported_callee_summary() -> None:
    summary = _summary(
        """
class Chain:
    def nested(self) -> int:
        return 1

    def apply(self) -> int:
        return self.nested()
""",
        "Chain.apply",
    )
    local_call = summary.local_calls[0]
    unsupported = replace(
        local_call.callee_summary,
        category=local_call.callee_summary.category.UNSUPPORTED,
    )

    with pytest.raises(ValueError, match="callee_summary"):
        replace(local_call, callee_summary=unsupported)


def test_transitive_observer_cost_cannot_be_forged() -> None:
    summary = _summary(
        """
class Reader:
    def read(self) -> int:
        return 1

    def inspect(self) -> int:
        self.read()
        return self.read()
""",
        "Reader.inspect",
    )
    forged_footprint = replace(
        summary.budget_footprint,
        observer_calls=0,
    )

    with pytest.raises(ValueError, match="observer_call_count|budget_footprint"):
        replace(
            summary,
            observer_call_count=0,
            budget_footprint=forged_footprint,
        )


def test_nested_receiver_proof_must_use_plan_module_identity() -> None:
    tree = ast.parse(_bounded_runtime_source())
    spec, reason = analyze_safe_custom_object_target(tree, "evaluate")
    assert reason is None and spec is not None
    context, context_reason = analyze_safe_object_setup_context(
        tree,
        module_identity="fixtures.module_binding",
        target_name="evaluate",
        custom_object_spec=spec,
    )
    assert context_reason is None and context is not None
    scenario = context.bind_scenario(
        Scenario(
            scenario_id="module-binding",
            name="module binding",
            path_index=1,
            priority_rank=1,
            priority_level="High",
            dqm_score=100.0,
            node_ids=(1,),
            edge_labels=(),
            contains_loop=False,
            contains_exception=False,
            description="module binding",
            keyword_arguments=tuple(
                {
                    context.state_analysis_name: True,
                    "mode": "",
                    **{
                        observer.analysis_name: (
                            "unit" if observer.return_type == "str" else 10
                        )
                        for observer in context.observers
                    },
                }.items()
            ),
            expected_result="unit",
        )
    )
    assert scenario.setup_plan is not None
    plan = scenario.setup_plan
    setup_call = plan.setup_calls[0]
    local_call = setup_call.method_summary.local_calls[0]
    forged_callee = replace(
        local_call.callee_summary,
        receiver=replace(
            local_call.callee_summary.receiver,
            module_identity="other.module",
        ),
    )
    forged_local_call = replace(local_call, callee_summary=forged_callee)
    forged_summary = replace(
        setup_call.method_summary,
        local_calls=(forged_local_call,),
    )
    forged_nested_slot = replace(
        plan.object_slots[1],
        blueprint=replace(
            plan.object_slots[1].blueprint,
            class_fingerprint=forged_callee.receiver.class_fingerprint,
        ),
    )

    with pytest.raises(SafeSetupPlanRejection) as error:
        SafeObjectSetupPlan(
            module_identity=plan.module_identity,
            target_identity=plan.target_identity,
            object_slots=(plan.object_slots[0], forged_nested_slot),
            setup_calls=(replace(setup_call, method_summary=forged_summary),),
            target_bindings=plan.target_bindings,
            budgets=plan.budgets,
        )

    assert (
        error.value.category
        is SafeSetupPlanRejectionCategory.UNSUPPORTED_METHOD_PROOF
    )
