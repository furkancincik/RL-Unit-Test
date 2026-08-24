from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from analyzer.safe_custom_object import (
    SafeObjectConstructionBlueprint,
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
