from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from analyzer.safe_custom_object import (
    SafeObjectConstructionBlueprint,
    UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER,
    analyze_safe_custom_object_target,
    normalized_custom_object_target,
)
from cfg.control_flow_graph import ControlFlowGraphBuilder
from generator.project_pytest_generator import ProjectPytestGenerator
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario
from models.project_coverage_result import ProjectTestCandidate
from services.source_analysis_orchestrator import SourceAnalysisOrchestrator
from services.real_rl_training_service import RealRLTrainingService


def _write_source(tmp_path: Path, source: str, name: str = "subject.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _scenario(**values: object) -> Scenario:
    return Scenario(
        scenario_id="stable_scenario_001",
        name="safe object",
        path_index=1,
        priority_rank=1,
        priority_level="High",
        dqm_score=1.0,
        node_ids=(1, 2),
        edge_labels=(None,),
        contains_loop=False,
        contains_exception=False,
        description="safe object scenario",
        keyword_arguments=tuple(values.items()),
        expected_result="ok",
    )


def test_analyzer_accepts_local_annotated_object_for_top_level_function(
    tmp_path: Path,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Parcel:\n"
        "    def __init__(self, weight: int = 4):\n"
        "        self.weight = weight\n\n"
        "def inspect(parcel: Parcel, threshold: int) -> str:\n"
        "    if parcel.weight > threshold:\n"
        "        return 'heavy'\n"
        "    return 'light'\n",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == "inspect"
    )

    assert target.is_supported is True
    assert target.parameters == ["parcel", "threshold"]
    assert target.parameter_types == {"parcel": "Parcel", "threshold": "int"}


def test_string_forward_annotation_resolves_only_to_local_top_level_class(
    tmp_path: Path,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Token:\n"
        "    def __init__(self, code: str = 'x'):\n"
        "        self.code = code\n\n"
        "def read(value: \"Token\") -> str:\n"
        "    return value.code\n",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == "read"
    )

    assert target.is_supported is True
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    spec, reason = analyze_safe_custom_object_target(tree, "read")
    assert reason is None
    assert spec is not None
    assert spec.object_parameters[0].class_name == "Token"


def test_simple_instance_method_accepts_annotated_custom_object_parameter(
    tmp_path: Path,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Reading:\n"
        "    def __init__(self, value: float):\n"
        "        self.value = value\n\n"
        "class Monitor:\n"
        "    def check(self, reading: Reading, limit: float) -> bool:\n"
        "        return reading.value > limit\n",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.qualified_name == "Monitor.check"
    )

    assert target.is_supported is True
    assert target.parameters == ["reading", "limit"]


@pytest.mark.parametrize(
    (
        "module_name",
        "class_name",
        "function_name",
        "parameter_name",
        "attribute_name",
        "literal",
    ),
    (
        ("structural_alpha", "Gauge", "classify", "subject", "level", 7),
        ("structural_beta", "Packet", "route", "payload", "weight", 19),
        ("structural_gamma", "Marker", "choose", "candidate", "rank", 31),
    ),
)
def test_untyped_parameter_infers_one_safe_structural_local_class(
    tmp_path: Path,
    module_name: str,
    class_name: str,
    function_name: str,
    parameter_name: str,
    attribute_name: str,
    literal: int,
) -> None:
    source_file = _write_source(
        tmp_path,
        f"class {class_name}:\n"
        f"    def __init__(self, seed: int = {literal}):\n"
        f"        self.{attribute_name} = seed\n\n"
        f"def {function_name}({parameter_name}, boundary: int) -> str:\n"
        f"    if {parameter_name}.{attribute_name} >= boundary:\n"
        "        return 'upper'\n"
        "    return 'lower'\n",
        f"{module_name}.py",
    )
    tree = ast.parse(source_file.read_text(encoding="utf-8"))

    spec, reason = analyze_safe_custom_object_target(tree, function_name)
    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == function_name
    )

    assert reason is None and spec is not None
    assert spec.object_parameters[0].parameter_name == parameter_name
    assert spec.object_parameters[0].class_name == class_name
    assert spec.object_parameters[0].resolution_kind == "STRUCTURAL_UNIQUE"
    assert target.is_supported is True
    assert target.parameter_types[parameter_name] == class_name


@pytest.mark.parametrize(
    ("first_name", "second_name", "attribute_name", "function_name"),
    (
        ("First", "Second", "signal", "inspect"),
        ("Primary", "Alternate", "weight", "evaluate"),
        ("North", "South", "level", "choose"),
    ),
)
def test_untyped_structural_parameter_rejects_ambiguous_safe_classes(
    first_name: str,
    second_name: str,
    attribute_name: str,
    function_name: str,
) -> None:
    tree = ast.parse(
        f"class {first_name}:\n"
        "    def __init__(self, value: int = 1):\n"
        f"        self.{attribute_name} = value\n\n"
        f"class {second_name}:\n"
        "    def __init__(self, value: int = 2):\n"
        f"        self.{attribute_name} = value\n\n"
        f"def {function_name}(subject) -> int:\n"
        f"    return subject.{attribute_name}\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, function_name)

    assert spec is None
    assert reason == (
        "Untyped structural object parameter has an ambiguous safe local "
        "class match."
    )


@pytest.mark.parametrize(
    ("class_name", "available_name", "missing_name", "function_name"),
    (
        ("Partial", "left", "right", "inspect"),
        ("Fragment", "mass", "volume", "measure"),
        ("Record", "rank", "score", "select"),
    ),
)
def test_untyped_structural_parameter_requires_a_safe_complete_match(
    class_name: str,
    available_name: str,
    missing_name: str,
    function_name: str,
) -> None:
    tree = ast.parse(
        f"class {class_name}:\n"
        "    def __init__(self, value: int = 1):\n"
        f"        self.{available_name} = value\n\n"
        f"def {function_name}(subject) -> int:\n"
        f"    return subject.{available_name} + subject.{missing_name}\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, function_name)

    assert spec is None
    assert reason is None


def test_instance_method_accepts_unique_untyped_structural_object_parameter(
    tmp_path: Path,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Reading:\n"
        "    def __init__(self, score: int = 3):\n"
        "        self.score = score\n\n"
        "class Evaluator:\n"
        "    def decide(self, candidate, limit: int) -> str:\n"
        "        if candidate.score > limit:\n"
        "            return 'above'\n"
        "        return 'other'\n",
        "structural_method.py",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.qualified_name == "Evaluator.decide"
    )

    assert target.is_supported is True
    assert target.parameters == ["candidate", "limit"]
    assert target.parameter_types == {"candidate": "Reading", "limit": "int"}


def test_untyped_structural_method_call_remains_controlled_unsupported() -> None:
    tree = ast.parse(
        "class Calculator:\n"
        "    def __init__(self):\n"
        "        self.entries = {}\n\n"
        "    def calculate(self) -> int:\n"
        "        return self.entries\n\n"
        "def handle(subject):\n"
        "    if not subject.entries:\n"
        "        return 0\n"
        "    return subject.calculate()\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "handle")

    assert reason is None
    assert spec is not None
    assert spec.object_parameters[0].resolution_kind == "STRUCTURAL_UNIQUE"
    normalized = ast.unparse(normalized_custom_object_target(spec))
    assert "if not {}" in normalized
    assert "subject.calculate()" not in normalized
    assert UNSUPPORTED_CUSTOM_OBJECT_METHOD_MARKER in normalized


@pytest.mark.parametrize(
    ("literal", "kind", "normalized_literal"),
    (
        ("{}", "EMPTY_DICT", "{}"),
        ("[]", "EMPTY_LIST", "[]"),
        ("()", "EMPTY_TUPLE", "()"),
    ),
)
def test_constructor_proven_empty_collection_state_is_normalized(
    literal: str,
    kind: str,
    normalized_literal: str,
) -> None:
    tree = ast.parse(
        "class Vessel:\n"
        "    def __init__(self):\n"
        f"        self.contents = {literal}\n\n"
        "def inspect(subject: Vessel) -> str:\n"
        "    if not subject.contents:\n"
        "        return 'empty'\n"
        "    return 'other'\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    initializer = spec.object_parameters[0].state_initializers[0]
    assert initializer.empty_collection_kind == kind
    assert initializer.uses_literal is False
    normalized = ast.unparse(normalized_custom_object_target(spec))
    assert f"if not {normalized_literal}" in normalized


def test_constructor_proven_empty_state_supports_unique_structural_object() -> None:
    tree = ast.parse(
        "class Archive:\n"
        "    def __init__(self):\n"
        "        self.records = {}\n\n"
        "    def calculate(self) -> int:\n"
        "        return len(self.records)\n\n"
        "def inspect(candidate) -> str:\n"
        "    if not candidate.records:\n"
        "        return 'vacant'\n"
        "    return candidate.calculate()\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    assert spec.object_parameters[0].class_name == "Archive"
    assert spec.object_parameters[0].resolution_kind == "STRUCTURAL_UNIQUE"
    assert (
        spec.object_parameters[0].state_initializers[0].empty_collection_kind
        == "EMPTY_DICT"
    )


def test_empty_collection_state_proof_fingerprint_is_stable_and_kind_sensitive() -> None:
    def proof_for(literal: str) -> tuple[str, str]:
        tree = ast.parse(
            "class State:\n"
            "    def __init__(self):\n"
            f"        self.payload = {literal}\n\n"
            "def inspect(subject: State) -> bool:\n"
            "    return not subject.payload\n"
        )
        spec, reason = analyze_safe_custom_object_target(tree, "inspect")
        assert reason is None and spec is not None
        initializer = spec.object_parameters[0].state_initializers[0]
        return initializer.proof_fingerprint, spec.object_parameters[0].class_fingerprint

    first_proof, first_class = proof_for("{}")
    equivalent_proof, equivalent_class = proof_for("{}")
    list_proof, list_class = proof_for("[]")

    assert first_proof == equivalent_proof
    assert first_class == equivalent_class
    assert first_proof != list_proof
    assert first_class != list_class


@pytest.mark.parametrize(
    "constructor_body",
    (
        "self.entries = {'seed': 1}",
        "if enabled:\n            self.entries = {}",
        "temporary = {}\n        self.entries = temporary",
        "self.entries = {value for value in ()}",
        "self.entries = factory()",
        "self.entries = {}\n        self.entries = []",
        "self.entries = {}\n        del self.entries",
    ),
)
def test_unsafe_empty_collection_constructor_proof_is_rejected(
    constructor_body: str,
) -> None:
    extra_parameter = "enabled: bool = True" if "enabled" in constructor_body else ""
    tree = ast.parse(
        "class UnsafeState:\n"
        f"    def __init__(self, {extra_parameter}):\n"
        f"        {constructor_body}\n\n"
        "def inspect(subject: UnsafeState) -> bool:\n"
        "    return not subject.entries\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert reason is not None


@pytest.mark.parametrize(
    "constructor_signature",
    (
        "entries: list = []",
        "entries: dict = {}",
        "entries: tuple = ()",
    ),
)
def test_constructor_parameter_or_mutable_default_is_not_empty_state_proof(
    constructor_signature: str,
) -> None:
    tree = ast.parse(
        "class UnsafeState:\n"
        f"    def __init__(self, {constructor_signature}):\n"
        "        self.entries = entries\n\n"
        "def inspect(subject: UnsafeState) -> bool:\n"
        "    return not subject.entries\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert reason is not None


@pytest.mark.parametrize(
    "source",
    (
        (
            "class BuiltinCall:\n"
            "    def __init__(self):\n"
            "        self.entries = list()\n\n"
            "def inspect(subject: BuiltinCall):\n"
            "    return not subject.entries\n"
        ),
        (
            "list = lambda: []\n\n"
            "class ShadowedCall:\n"
            "    def __init__(self):\n"
            "        self.entries = list()\n\n"
            "def inspect(subject: ShadowedCall):\n"
            "    return not subject.entries\n"
        ),
    ),
)
def test_empty_collection_constructor_calls_remain_controlled_unsupported(
    source: str,
) -> None:
    spec, reason = analyze_safe_custom_object_target(ast.parse(source), "inspect")

    assert spec is None
    assert "constructor" in (reason or "").lower()


@pytest.mark.parametrize(
    "target_body",
    (
        "return subject.entries",
        "return bool(subject.entries)",
        "return len(subject.entries) == 0",
        "return subject.entries == {}",
        "subject.entries = {}\n    return True",
        "del subject.entries\n    return True",
        "subject.entries.append(1)\n    return True",
        "subject.entries['key'] = 1\n    return True",
    ),
)
def test_empty_collection_target_uses_outside_safe_not_are_rejected(
    target_body: str,
) -> None:
    tree = ast.parse(
        "class State:\n"
        "    def __init__(self):\n"
        "        self.entries = []\n\n"
        "def inspect(subject: State):\n"
        f"    {target_body}\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert reason is not None


@pytest.mark.parametrize(
    "class_header",
    (
        "class State(Base):",
        "class State(metaclass=Meta):",
    ),
)
def test_empty_collection_candidate_rejects_inheritance_and_metaclass(
    class_header: str,
) -> None:
    tree = ast.parse(
        f"{class_header}\n"
        "    def __init__(self):\n"
        "        self.entries = {}\n\n"
        "def inspect(subject: State):\n"
        "    return not subject.entries\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert "inheritance" in (reason or "").lower()


def test_empty_collection_candidate_rejects_property_descriptors() -> None:
    tree = ast.parse(
        "class State:\n"
        "    def __init__(self):\n"
        "        self.entries = {}\n\n"
        "    @property\n"
        "    def visible(self):\n"
        "        return self.entries\n\n"
        "def inspect(subject: State):\n"
        "    return not subject.entries\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert "descriptor" in (reason or "").lower()


def test_explicit_local_annotation_takes_precedence_over_structural_candidates() -> None:
    tree = ast.parse(
        "class Chosen:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n\n"
        "class Other:\n"
        "    def __init__(self, value: int = 2):\n"
        "        self.signal = value\n\n"
        "def inspect(subject: Chosen) -> int:\n"
        "    return subject.signal\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    assert spec.object_parameters[0].class_name == "Chosen"
    assert spec.object_parameters[0].resolution_kind == "ANNOTATED"


def test_inferred_primitive_evidence_takes_precedence_over_structural_shape() -> None:
    tree = ast.parse(
        "class StringLike:\n"
        "    def transform(self):\n"
        "        return 'changed'\n\n"
        "def inspect(value):\n"
        "    if value == 'ready':\n"
        "        return value.transform()\n"
        "    return value\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert reason is None


def test_unsafe_structural_candidate_does_not_make_safe_match_ambiguous() -> None:
    tree = ast.parse(
        "class SafeCandidate:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n\n"
        "class UnsafeCandidate:\n"
        "    def __init__(self, value: int = 2):\n"
        "        self.signal = transform(value)\n\n"
        "def inspect(subject) -> int:\n"
        "    return subject.signal\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    assert spec.object_parameters[0].class_name == "SafeCandidate"


@pytest.mark.parametrize(
    ("body", "reason"),
    (
        (
            "probe = subject.signal\n    return subject",
            "direct instance interface",
        ),
        (
            "probe = subject.signal\n    alias = subject\n    return alias",
            "direct instance interface",
        ),
        (
            "def nested():\n        return subject.signal\n    return nested()",
            "nested scopes",
        ),
        (
            "probe = subject.signal\n    return subject[0]",
            "direct instance interface",
        ),
    ),
)
def test_untyped_structural_inference_rejects_non_direct_or_nested_use(
    body: str,
    reason: str,
) -> None:
    tree = ast.parse(
        "class Candidate:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n\n"
        "def inspect(subject):\n"
        f"    {body}\n"
    )

    spec, rejection = analyze_safe_custom_object_target(tree, "inspect")

    assert spec is None
    assert reason in (rejection or "").lower()


@pytest.mark.parametrize("statement", ("subject.signal = 2", "del subject.signal"))
def test_untyped_structural_inference_rejects_attribute_write_and_delete(
    statement: str,
) -> None:
    tree = ast.parse(
        "class Candidate:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n\n"
        "def mutate(subject):\n"
        f"    {statement}\n"
        "    return 0\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "mutate")

    assert spec is None
    assert "write or delete" in (reason or "").lower()


def test_structural_callable_signature_selects_only_compatible_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        "class Compatible:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n"
        "    def calculate(self, scale: int, *, enabled: bool = True):\n"
        "        return self.signal * scale\n\n"
        "class Incompatible:\n"
        "    def __init__(self, value: int = 2):\n"
        "        self.signal = value\n"
        "    def calculate(self, scale: int, required: int):\n"
        "        return self.signal * scale + required\n\n"
        "def inspect(subject, factor: int):\n"
        "    if subject.signal > 0:\n"
        "        return subject.calculate(factor, enabled=False)\n"
        "    return 0\n"
    )
    monkeypatch.setattr(
        "analyzer.safe_custom_object._validate_target_usage",
        lambda target, parameter: None,
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    assert spec.object_parameters[0].class_name == "Compatible"
    assert spec.object_parameters[0].resolution_kind == "STRUCTURAL_UNIQUE"


@pytest.mark.parametrize(
    "source",
    (
        (
            "class StateOnly:\n"
            "    def __init__(self, value: int = 1):\n"
            "        self.action = value\n\n"
            "def inspect(subject):\n"
            "    return subject.action()\n"
        ),
        (
            "class MethodOnly:\n"
            "    def action(self):\n"
            "        return 1\n\n"
            "def inspect(subject):\n"
            "    return subject.action\n"
        ),
    ),
)
def test_structural_attribute_and_callable_evidence_are_not_interchangeable(
    source: str,
) -> None:
    spec, reason = analyze_safe_custom_object_target(ast.parse(source), "inspect")

    assert spec is None
    assert reason is None


def test_untyped_parameter_without_attribute_evidence_is_not_object_inference() -> None:
    tree = ast.parse(
        "class Candidate:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n\n"
        "def calculate(value):\n"
        "    return value + 1\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "calculate")

    assert spec is None
    assert reason is None


def test_unmatched_structural_parameter_does_not_hide_later_safe_object() -> None:
    tree = ast.parse(
        "class Known:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.signal = value\n\n"
        "def inspect(unknown, known: Known) -> int:\n"
        "    if unknown.missing:\n"
        "        return 0\n"
        "    return known.signal\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    assert tuple(
        (parameter.parameter_name, parameter.class_name)
        for parameter in spec.object_parameters
    ) == (("known", "Known"),)


def test_multiple_untyped_structural_parameters_are_inferred_independently() -> None:
    tree = ast.parse(
        "class LeftShape:\n"
        "    def __init__(self, value: int = 1):\n"
        "        self.left_signal = value\n\n"
        "class RightShape:\n"
        "    def __init__(self, value: int = 2):\n"
        "        self.right_signal = value\n\n"
        "def compare(first, second) -> bool:\n"
        "    return first.left_signal > second.right_signal\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "compare")

    assert reason is None and spec is not None
    assert tuple(
        (parameter.parameter_name, parameter.class_name)
        for parameter in spec.object_parameters
    ) == (("first", "LeftShape"), ("second", "RightShape"))


def test_custom_object_constructor_supports_primitive_defaults_and_inference() -> None:
    tree = ast.parse(
        "class Measure:\n"
        "    def __init__(self, count=7, label: str = 'a', enabled=False):\n"
        "        self.count = count\n"
        "        self.label = label\n"
        "        self.enabled = enabled\n\n"
        "def inspect(value: Measure) -> bool:\n"
        "    return value.enabled\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None
    assert spec is not None
    assert tuple(
        parameter.type_name
        for parameter in spec.object_parameters[0].constructor_parameters
    ) == ("int", "str", "bool")


def test_multiple_custom_object_parameters_are_independent_and_bounded() -> None:
    supported_tree = ast.parse(
        "class Left:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "class Right:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "def compare(left: Left, right: Right) -> bool:\n"
        "    return left.value > right.value\n"
    )
    spec, reason = analyze_safe_custom_object_target(supported_tree, "compare")
    assert reason is None
    assert spec is not None
    assert len(spec.object_parameters) == 2
    assert len(set(spec.analysis_parameter_names)) == len(spec.analysis_parameter_names)

    class_blocks = []
    parameters = []
    for index in range(5):
        class_blocks.append(
            f"class Shape{index}:\n"
            "    def __init__(self, value: int):\n"
            "        self.value = value\n"
        )
        parameters.append(f"item{index}: Shape{index}")
    limited_tree = ast.parse(
        "\n".join(class_blocks)
        + "\ndef inspect("
        + ", ".join(parameters)
        + ") -> int:\n    return 1\n"
    )
    rejected, rejection = analyze_safe_custom_object_target(limited_tree, "inspect")
    assert rejected is None
    assert "limit" in (rejection or "").lower()


def test_synthetic_constructor_inputs_avoid_real_parameter_name_collisions() -> None:
    tree = ast.parse(
        "class Local:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "def inspect(__custom_object_001_001: int, subject: Local) -> int:\n"
        "    return subject.value + __custom_object_001_001\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "inspect")

    assert reason is None and spec is not None
    assert len(set(spec.analysis_parameter_names)) == 2
    assert spec.analysis_parameter_names[0] == "__custom_object_001_001"
    assert spec.analysis_parameter_names[1] != "__custom_object_001_001"


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            "class Child(Base):\n"
            "    def __init__(self, value: int):\n"
            "        self.value = value\n\n"
            "def use(value: Child):\n    return value.value\n",
            "inheritance",
        ),
        (
            "class Item:\n"
            "    def __init__(self, value: int):\n"
            "        self.value = helper(value)\n\n"
            "def use(value: Item):\n    return value.value\n",
            "constructor",
        ),
        (
            "class Item:\n"
            "    def __init__(self, child: Item):\n"
            "        self.child = child\n\n"
            "def use(value: Item):\n    return 1\n",
            "primitive",
        ),
        (
            "class Item:\n"
            "    def __new__(cls):\n"
            "        return super().__new__(cls)\n\n"
            "def use(value: Item):\n    return 1\n",
            "__new__",
        ),
        (
            "def use(value: package.Item):\n    return 1\n",
            "attribute",
        ),
    ],
)
def test_unsafe_or_unresolved_custom_objects_are_controlled_unsupported(
    tmp_path: Path,
    source: str,
    reason: str,
) -> None:
    target = next(
        item
        for item in PythonAnalyzer().analyze_file(
            _write_source(tmp_path, source)
        ).functions
        if item.name == "use"
    )

    assert target.is_supported is False
    assert reason in (target.unsupported_reason or "").lower()


def test_normalized_target_replaces_object_state_with_primitive_inputs() -> None:
    tree = ast.parse(
        "class Gauge:\n"
        "    def __init__(self, level: int):\n"
        "        self.level = level\n"
        "        self.active = True\n\n"
        "def classify(gauge: Gauge, boundary: int) -> str:\n"
        "    if gauge.active and gauge.level >= boundary:\n"
        "        return 'on'\n"
        "    return 'off'\n"
    )
    spec, reason = analyze_safe_custom_object_target(tree, "classify")
    assert reason is None and spec is not None
    node = normalized_custom_object_target(spec)
    text = ast.unparse(node)

    assert "gauge." not in text
    assert "gauge" not in [argument.arg for argument in node.args.args]
    assert "True" in text
    assert any(name.startswith("__custom_object_") for name in spec.analysis_parameter_names)


def test_cfg_uses_normalized_custom_object_state(tmp_path: Path) -> None:
    source_file = _write_source(
        tmp_path,
        "class Cell:\n"
        "    def __init__(self, charge: int):\n"
        "        self.charge = charge\n\n"
        "def classify(cell: Cell) -> str:\n"
        "    if cell.charge > 0:\n"
        "        return 'positive'\n"
        "    return 'other'\n",
    )

    graph = next(
        graph
        for graph in ControlFlowGraphBuilder().build_from_file(source_file)
        if graph.function_name == "classify"
    )

    labels = tuple(node.label for node in graph.nodes)
    assert all("cell." not in label for label in labels)
    assert any("__custom_object_" in label for label in labels)


def test_blueprint_is_immutable_stable_and_hides_constructor_values() -> None:
    first = SafeObjectConstructionBlueprint(
        module_identity="package.local_module",
        class_name="LocalType",
        constructor_arguments=(("quantity", 19), ("enabled", True)),
    )
    second = SafeObjectConstructionBlueprint(
        module_identity="package.local_module",
        class_name="LocalType",
        constructor_arguments=(("quantity", 19), ("enabled", True)),
    )

    assert first.fingerprint == second.fingerprint
    assert "19" not in repr(first)
    assert "constructor_arguments" not in repr(first)
    with pytest.raises(FrozenInstanceError):
        first.depth = 2  # type: ignore[misc]


def test_fingerprint_module_identity_is_stable_and_separates_local_targets() -> None:
    first = SafeObjectConstructionBlueprint(
        module_identity="package.first_module",
        class_name="Shared",
        constructor_arguments=(("value", 19),),
    )
    equivalent = SafeObjectConstructionBlueprint(
        module_identity="package.first_module",
        class_name="Shared",
        constructor_arguments=(("value", 19),),
    )
    other_module = SafeObjectConstructionBlueprint(
        module_identity="package.second_module",
        class_name="Shared",
        constructor_arguments=(("value", 19),),
    )
    other_class = SafeObjectConstructionBlueprint(
        module_identity="package.first_module",
        class_name="Alternate",
        constructor_arguments=(("value", 19),),
    )
    other_value = SafeObjectConstructionBlueprint(
        module_identity="package.first_module",
        class_name="Shared",
        constructor_arguments=(("value", 23),),
    )

    assert first.fingerprint == equivalent.fingerprint
    assert first.fingerprint != other_module.fingerprint
    assert first.fingerprint != other_class.fingerprint
    assert first.fingerprint != other_value.fingerprint
    assert "package.first_module" not in repr(first)


@pytest.mark.parametrize(
    "module_identity",
    (
        "",
        " package.module",
        "package.module ",
        "package/module",
        "package\\module",
        "C:\\temporary\\module.py",
        ".package.module",
        "package..module",
        "package-module",
    ),
)
def test_blueprint_rejects_missing_or_path_based_module_identity(
    module_identity: str,
) -> None:
    with pytest.raises(ValueError, match="module_identity"):
        SafeObjectConstructionBlueprint(
            module_identity=module_identity,
            class_name="Shared",
        )


def test_scenario_binding_uses_validated_module_identity_without_cross_module_collision() -> None:
    source = (
        "class Shared:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "def inspect(subject: Shared) -> int:\n"
        "    return subject.value\n"
    )
    first_spec, first_reason = analyze_safe_custom_object_target(
        ast.parse(source),
        "inspect",
    )
    second_spec, second_reason = analyze_safe_custom_object_target(
        ast.parse(source),
        "inspect",
    )
    assert first_reason is None and first_spec is not None
    assert second_reason is None and second_spec is not None
    synthetic = first_spec.analysis_parameter_names[0]
    scenario = _scenario(**{synthetic: 31})

    first = first_spec.bind_scenario(
        scenario,
        module_identity="first_module",
    )
    second = second_spec.bind_scenario(
        scenario,
        module_identity="second_module",
    )
    first_blueprint = first.keyword_argument_dict["subject"]
    second_blueprint = second.keyword_argument_dict["subject"]

    assert first_blueprint.module_identity == "first_module"
    assert second_blueprint.module_identity == "second_module"
    assert first_blueprint.fingerprint != second_blueprint.fingerprint
    assert first_blueprint.constructor_arguments == second_blueprint.constructor_arguments


def test_binding_creates_independent_blueprints_without_runtime_instances() -> None:
    tree = ast.parse(
        "class Box:\n"
        "    def __init__(self, size: int):\n"
        "        self.size = size\n\n"
        "def inspect(box: Box) -> int:\n"
        "    box.size += 1\n"
        "    return box.size\n"
    )
    spec, reason = analyze_safe_custom_object_target(tree, "inspect")
    assert reason is None and spec is not None
    synthetic = spec.analysis_parameter_names[0]

    first = spec.bind_scenario(
        _scenario(**{synthetic: 3}),
        module_identity="package.box_module",
    )
    second = spec.bind_scenario(
        _scenario(**{synthetic: 3}),
        module_identity="package.box_module",
    )
    first_blueprint = first.keyword_argument_dict["box"]
    second_blueprint = second.keyword_argument_dict["box"]

    assert isinstance(first_blueprint, SafeObjectConstructionBlueprint)
    assert first_blueprint is not second_blueprint
    assert first_blueprint == second_blueprint
    assert first.scenario_id == second.scenario_id


def test_concrete_validation_materializes_fresh_instance_per_scenario(
    tmp_path: Path,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Mutable:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "def advance(subject: Mutable) -> int:\n"
        "    subject.value += 1\n"
        "    return subject.value\n",
        "isolated_runtime.py",
    )
    blueprint = SafeObjectConstructionBlueprint(
        module_identity="isolated_runtime",
        class_name="Mutable",
        constructor_arguments=(("value", 8),),
    )
    scenarios = (
        replace(_scenario(subject=blueprint), expected_result=9),
        replace(
            _scenario(subject=blueprint),
            scenario_id="stable_scenario_002",
            path_index=2,
            priority_rank=2,
            expected_result=9,
        ),
    )

    accepted = RealRLTrainingService()._filter_executable_scenarios(
        source_file=source_file,
        function_name="advance",
        scenarios=scenarios,
    )

    assert accepted == scenarios


def test_generated_pytest_constructs_fresh_object_for_top_level_function() -> None:
    blueprint = SafeObjectConstructionBlueprint(
        module_identity="package.module",
        class_name="Payload",
        constructor_arguments=(("value", 5),),
    )
    scenario = _scenario(payload=blueprint)

    source = PytestGenerator().generate("package.module", "inspect", (scenario,))

    assert "from package.module import Payload, inspect" in source
    assert "payload_object = Payload(value=5)" in source
    assert "result = inspect(payload=payload_object)" in source
    assert "SafeObjectConstructionBlueprint" not in source


def test_generated_pytest_constructs_object_for_instance_method_parameter() -> None:
    blueprint = SafeObjectConstructionBlueprint(
        module_identity="package.module",
        class_name="Reading",
        constructor_arguments=(("value", 2.5),),
    )
    scenario = replace(
        _scenario(reading=blueprint),
        constructor_arguments=(("enabled", True),),
        target_class_name="Monitor",
    )

    source = PytestGenerator().generate("package.module", "check", (scenario,))

    assert "from package.module import Monitor, Reading" in source
    assert "target = Monitor(enabled=True)" in source
    assert "reading_object = Reading(value=2.5)" in source
    assert "target.check(reading=reading_object)" in source


def test_project_pytest_uses_blueprint_without_repr_or_memory_address(
    tmp_path: Path,
) -> None:
    source_file = _write_source(tmp_path, "def inspect(value):\n    return 1\n")
    scenario = _scenario(
        value=SafeObjectConstructionBlueprint(
            module_identity="module",
            class_name="Sample",
            constructor_arguments=(("code", "safe"),),
        )
    )
    candidate = ProjectTestCandidate(
        project_test_id="module::inspect::001",
        relative_module_path="module.py",
        module_path="module",
        function_name="inspect",
        function_start_line=1,
        function_end_line=2,
        source_file=source_file,
        import_root=tmp_path,
        scenario=scenario,
        original_order=1,
    )

    generated = ProjectPytestGenerator().generate((candidate,))

    assert "Sample" in generated
    assert "_project_object_class_0001_001(code='safe')" in generated
    assert "SafeObjectConstructionBlueprint" not in generated
    assert "object at 0x" not in generated


@pytest.mark.parametrize(
    ("module_name", "class_name", "function_name", "parameter_name", "attribute"),
    [
        ("alpha_case", "Capsule", "evaluate", "subject", "magnitude"),
        ("beta_case", "Packet", "decide", "candidate", "quantity"),
        ("gamma_case", "Marker", "select", "record", "level"),
    ],
)
def test_custom_object_policy_is_identifier_independent(
    tmp_path: Path,
    module_name: str,
    class_name: str,
    function_name: str,
    parameter_name: str,
    attribute: str,
) -> None:
    source_file = _write_source(
        tmp_path,
        f"class {class_name}:\n"
        "    def __init__(self, signal: int = 13):\n"
        f"        self.{attribute} = signal\n\n"
        f"def {function_name}({parameter_name}: {class_name}) -> str:\n"
        f"    if {parameter_name}.{attribute} >= 17:\n"
        "        return 'upper'\n"
        "    return 'lower'\n",
        f"{module_name}.py",
    )

    target = next(
        item
        for item in PythonAnalyzer().analyze_file(source_file).functions
        if item.name == function_name
    )
    assert target.is_supported is True


def test_real_pipeline_runs_custom_object_pytest_coverage_greedy_and_rl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Container:\n"
        "    def __init__(self, amount: int = 2):\n"
        "        self.amount = amount\n\n"
        "def classify(container: Container, boundary: int) -> str:\n"
        "    if container.amount >= boundary:\n"
        "        return 'enough'\n"
        "    return 'short'\n",
        "dynamic_custom.py",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="dynamic_custom",
        function_name="classify",
        all_functions=False,
        output_root=tmp_path / "output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        timeout_seconds=30.0,
        per_function_timeout_seconds=60.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status.value == "COMPLETED"
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.covered_lines
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert function_result.diagnostic is not None
    assert function_result.diagnostic.funnel.rl_executed_test_count
    public_payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "constructor_arguments" not in public_payload
    assert "keyword_arguments" not in public_payload
    assert "object at 0x" not in public_payload
    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "Container(amount=" in generated
    assert "classify(container=" in generated


def test_real_pipeline_runs_unique_untyped_structural_object_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Register:\n"
        "    def __init__(self, value: int = 5):\n"
        "        self.value = value\n\n"
        "def evaluate(subject, boundary: int) -> str:\n"
        "    if subject.value >= boundary:\n"
        "        return 'enough'\n"
        "    return 'short'\n",
        "structural_dynamic.py",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="structural_dynamic",
        function_name="evaluate",
        all_functions=False,
        output_root=tmp_path / "structural_output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=43,
        timeout_seconds=30.0,
        per_function_timeout_seconds=60.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status.value == "COMPLETED"
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.covered_lines
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert function_result.diagnostic is not None
    assert function_result.diagnostic.funnel.rl_executed_test_count
    assert result.coverage_candidates
    assert all(
        isinstance(
            candidate.scenario.keyword_argument_dict["subject"],
            SafeObjectConstructionBlueprint,
        )
        for candidate in result.coverage_candidates
    )
    public_payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "constructor_arguments" not in public_payload
    assert "keyword_arguments" not in public_payload
    assert "__custom_object_" not in public_payload
    assert "resolution_kind" not in public_payload
    assert "STRUCTURAL_UNIQUE" not in public_payload
    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from structural_dynamic import Register, evaluate" in generated
    assert "subject_object = Register(value=" in generated
    assert "evaluate(subject=subject_object" in generated


def test_real_pipeline_separates_same_named_local_classes_by_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        "class Shared:\n"
        "    def __init__(self, value: int = 3):\n"
        "        self.value = value\n\n"
        "def inspect(subject: Shared) -> str:\n"
        "    if subject.value > 0:\n"
        "        return 'positive'\n"
        "    return 'other'\n"
    )
    first_source = _write_source(tmp_path, source, "first_module.py")
    second_source = _write_source(tmp_path, source, "second_module.py")
    monkeypatch.chdir(tmp_path)

    def run(source_file: Path, module_path: str, output_name: str):
        return SourceAnalysisOrchestrator().run(
            source_file=source_file,
            module_path=module_path,
            function_name="inspect",
            all_functions=False,
            output_root=tmp_path / output_name,
            max_visits_per_node=2,
            episode_count=1,
            epsilon=0.0,
            learning_rate=0.5,
            discount_factor=0.9,
            random_seed=13,
            timeout_seconds=30.0,
            per_function_timeout_seconds=60.0,
            run_greedy_baseline=False,
        )

    first_result = run(first_source, "first_module", "first_output")
    second_result = run(second_source, "second_module", "second_output")

    def blueprints_by_arguments(result: object) -> dict[tuple[tuple[str, object], ...], object]:
        candidates = result.coverage_candidates  # type: ignore[attr-defined]
        return {
            blueprint.constructor_arguments: blueprint
            for candidate in candidates
            for blueprint in candidate.scenario.keyword_argument_dict.values()
            if isinstance(blueprint, SafeObjectConstructionBlueprint)
        }

    first_blueprints = blueprints_by_arguments(first_result)
    second_blueprints = blueprints_by_arguments(second_result)
    assert first_blueprints.keys() == second_blueprints.keys()
    assert first_blueprints
    assert all(
        first_blueprints[arguments].module_identity == "first_module"
        and second_blueprints[arguments].module_identity == "second_module"
        and first_blueprints[arguments].fingerprint
        != second_blueprints[arguments].fingerprint
        for arguments in first_blueprints
    )

    first_generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in first_result.function_results[0].output_directory.rglob(
            "test_*.py"
        )
    )
    second_generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in second_result.function_results[0].output_directory.rglob(
            "test_*.py"
        )
    )
    assert "from first_module import Shared, inspect" in first_generated
    assert "from second_module import Shared, inspect" in second_generated


def test_real_instance_method_pipeline_accepts_custom_object_method_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Reading:\n"
        "    def __init__(self, score: int = 1):\n"
        "        self.score = score\n\n"
        "class Evaluator:\n"
        "    def decide(self, reading: Reading, limit: int) -> str:\n"
        "        if reading.score > limit:\n"
        "            return 'above'\n"
        "        return 'other'\n",
        "method_custom.py",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="method_custom",
        function_name="Evaluator.decide",
        all_functions=False,
        output_root=tmp_path / "method_output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=7,
        timeout_seconds=30.0,
        per_function_timeout_seconds=60.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status.value == "COMPLETED"
    assert function_result.scenario_pool_coverage is not None
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert result.coverage_candidates
    assert all(
        isinstance(
            candidate.scenario.keyword_argument_dict["reading"],
            SafeObjectConstructionBlueprint,
        )
        for candidate in result.coverage_candidates
    )
    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from method_custom import Evaluator, Reading" in generated
    assert "reading_object = Reading(score=" in generated
    assert "target = Evaluator()" in generated
    assert "target.decide(reading=reading_object" in generated


def test_real_instance_method_pipeline_accepts_unique_untyped_object_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Signal:\n"
        "    def __init__(self, level: int = 2):\n"
        "        self.level = level\n\n"
        "class Judge:\n"
        "    def choose(self, candidate, limit: int) -> str:\n"
        "        if candidate.level > limit:\n"
        "            return 'above'\n"
        "        return 'other'\n",
        "structural_method_dynamic.py",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="structural_method_dynamic",
        function_name="Judge.choose",
        all_functions=False,
        output_root=tmp_path / "structural_method_output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=17,
        timeout_seconds=30.0,
        per_function_timeout_seconds=60.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status.value == "COMPLETED"
    assert function_result.scenario_pool_coverage is not None
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert function_result.diagnostic is not None
    assert function_result.diagnostic.funnel.rl_executed_test_count
    assert result.coverage_candidates
    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from structural_method_dynamic import Judge, Signal" in generated
    assert "candidate_object = Signal(level=" in generated
    assert "target = Judge()" in generated
    assert "target.choose(candidate=candidate_object" in generated


@pytest.mark.parametrize(
    ("module_name", "class_name", "function_name", "parameter_name", "attribute", "literal"),
    (
        ("empty_dict_case", "Ledger", "inspect", "subject", "records", "{}"),
        ("empty_list_case", "QueueState", "examine", "candidate", "entries", "[]"),
        ("empty_tuple_case", "Snapshot", "review", "payload", "tokens", "()"),
    ),
)
def test_constructor_proven_empty_policy_is_identifier_and_kind_independent(
    module_name: str,
    class_name: str,
    function_name: str,
    parameter_name: str,
    attribute: str,
    literal: str,
) -> None:
    tree = ast.parse(
        f"class {class_name}:\n"
        "    def __init__(self):\n"
        f"        self.{attribute} = {literal}\n\n"
        "    def compute(self) -> int:\n"
        f"        return len(self.{attribute})\n\n"
        f"def {function_name}({parameter_name}) -> str:\n"
        f"    if not {parameter_name}.{attribute}:\n"
        f"        return {module_name!r}\n"
        f"    return {parameter_name}.compute()\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, function_name)

    assert reason is None and spec is not None
    assert spec.object_parameters[0].resolution_kind == "STRUCTURAL_UNIQUE"
    assert spec.object_parameters[0].state_initializers[0].proof_fingerprint


def test_real_pipeline_keeps_empty_early_path_and_rejects_late_method_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = _write_source(
        tmp_path,
        "class Ledger:\n"
        "    def __init__(self):\n"
        "        self.records = {}\n\n"
        "    def calculate(self) -> int:\n"
        "        return len(self.records)\n\n"
        "def inspect(subject, invoke: bool) -> str | int:\n"
        "    if invoke:\n"
        "        subject.calculate()\n"
        "        return 'invoked'\n"
        "    if not subject.records:\n"
        "        return 'empty'\n"
        "    return 'other'\n",
        "path_scoped_empty.py",
    )
    monkeypatch.chdir(tmp_path)

    result = SourceAnalysisOrchestrator().run(
        source_file=source_file,
        module_path="path_scoped_empty",
        function_name="inspect",
        all_functions=False,
        output_root=tmp_path / "path_scoped_output",
        max_visits_per_node=2,
        episode_count=1,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=29,
        timeout_seconds=30.0,
        per_function_timeout_seconds=60.0,
        run_greedy_baseline=True,
    )

    function_result = result.function_results[0]
    assert function_result.status.value in {"COMPLETED", "PARTIAL"}
    assert function_result.scenario_count >= 1
    assert function_result.scenario_pool_coverage is not None
    assert function_result.scenario_pool_coverage.covered_lines
    assert function_result.minimization_result is not None
    assert function_result.minimization_result.coverage_preserved is True
    assert function_result.diagnostic is not None
    assert function_result.diagnostic.funnel.rl_executed_test_count >= 1
    assert any(
        count > 0
        for category, count in function_result.diagnostic.scenario_rejection_counts
        if category in {"UNSUPPORTED_EXPECTED_RESULT", "UNSUPPORTED_INPUT_SYNTHESIS"}
    )
    assert result.coverage_candidates
    assert all(
        isinstance(
            candidate.scenario.keyword_argument_dict["subject"],
            SafeObjectConstructionBlueprint,
        )
        for candidate in result.coverage_candidates
    )
    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in function_result.output_directory.rglob("test_*.py")
    )
    assert "from path_scoped_empty import Ledger, inspect" in generated
    assert "subject_object = Ledger()" in generated
    assert "inspect(subject=subject_object, invoke=False)" in generated
    assert "assert result == 'empty'" in generated
    public_payload = json.dumps(result.to_dict(), ensure_ascii=False)
    for secret in (
        "records",
        "EMPTY_DICT",
        "proof_fingerprint",
        "constructor_arguments",
        "keyword_arguments",
        "__custom_object_",
    ):
        assert secret not in public_payload


def test_instance_method_target_accepts_constructor_proven_object_parameter() -> None:
    tree = ast.parse(
        "class Bucket:\n"
        "    def __init__(self):\n"
        "        self.values = []\n\n"
        "class Inspector:\n"
        "    def inspect(self, subject: Bucket) -> str:\n"
        "        if not subject.values:\n"
        "            return 'empty'\n"
        "        return 'other'\n"
    )

    spec, reason = analyze_safe_custom_object_target(tree, "Inspector.inspect")

    assert reason is None and spec is not None
    normalized = normalized_custom_object_target(spec)
    assert [argument.arg for argument in normalized.args.args] == ["self"]
    assert "if not []" in ast.unparse(normalized)


def test_static_analyzer_never_executes_constructor_or_target(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed.txt"
    source_file = _write_source(
        tmp_path,
        "from pathlib import Path\n\n"
        "class SafeShape:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n\n"
        "def inspect(shape: SafeShape) -> int:\n"
        f"    Path({str(marker)!r}).write_text('executed')\n"
        "    return shape.value\n",
    )

    PythonAnalyzer().analyze_file(source_file)

    assert marker.exists() is False


@pytest.mark.parametrize("error_type", (AssertionError, TypeError, RuntimeError))
def test_unexpected_programming_errors_are_not_normalized_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    tree = ast.parse(
        "class Value:\n"
        "    def __init__(self, amount: int):\n"
        "        self.amount = amount\n\n"
        "def inspect(value: Value):\n"
        "    return value.amount\n"
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise error_type("programming failure")

    monkeypatch.setattr(
        "analyzer.safe_custom_object._analyze_safe_class",
        fail,
    )
    with pytest.raises(error_type, match="programming failure"):
        analyze_safe_custom_object_target(tree, "inspect")
