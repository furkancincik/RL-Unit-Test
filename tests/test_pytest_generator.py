from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from analyzer.safe_custom_object import SafeObjectConstructionBlueprint
from analyzer.safe_method_summary import analyze_safe_method_summaries
from generator.pytest_generator import PytestGenerator
from generator.project_pytest_generator import ProjectPytestGenerator
from generator.safe_method_setup_plan import (
    SafeObjectSetupPlan,
    SafeObjectSlot,
    SafeSetupCall,
    SafeTargetBinding,
    SafeTypedArgumentBinding,
)
from generator.scenario_generator import Scenario
from models.project_coverage_result import ProjectTestCandidate


def create_scenario(
    *,
    scenario_id: str = "calculate_score_scenario_001",
    path_index: int = 1,
    priority_rank: int = 1,
    priority_level: str = "High",
    dqm_score: float = 100.0,
    keyword_arguments: tuple[
        tuple[str, object],
        ...,
    ] = (("score", 85),),
    expected_result: object = "Başarılı",
    expected_exception: str | None = None,
) -> Scenario:
    """PytestGenerator testlerinde kullanılacak Scenario üretir."""
    return Scenario(
        scenario_id=scenario_id,
        name="calculate_score yürütme yolu",
        path_index=path_index,
        priority_rank=priority_rank,
        priority_level=priority_level,
        dqm_score=dqm_score,
        node_ids=(1, 3, 4, 2),
        edge_labels=(None, "True", None),
        contains_loop=False,
        contains_exception=(
            expected_exception is not None
        ),
        description=(
            "calculate_score fonksiyonu için test senaryosu."
        ),
        keyword_arguments=keyword_arguments,
        expected_result=expected_result,
        expected_exception=expected_exception,
    )


def test_generate_creates_valid_python_code() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(),
        ],
    )

    ast.parse(source_code)


def test_generate_imports_target_function() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(),
        ],
    )

    assert (
        "from datasets.sample_code "
        "import calculate_score"
    ) in source_code


def test_generate_creates_real_function_call() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(
                keyword_arguments=(
                    ("score", 85),
                ),
            ),
        ],
    )

    assert (
        "result = calculate_score(score=85)"
        in source_code
    )


def create_runtime_setup_plan(*, receiver_slot_id: str | None = None) -> SafeObjectSetupPlan:
    """Gerçek constructor ve ordered setup replay testi için plan."""
    module_identity = "example.runtime_setup"
    summary = analyze_safe_method_summaries(
        ast.parse(
            """
class Bucket:
    def __init__(self):
        self.total = 0

    def add(self, amount: int) -> None:
        self.total += amount
"""
        ),
        module_identity=module_identity,
    ).summary_for("Bucket.add")
    root = SafeObjectSlot(
        slot_id="root",
        blueprint=SafeObjectConstructionBlueprint(
            module_identity=module_identity,
            class_name="Bucket",
            class_fingerprint=summary.receiver.class_fingerprint,
        ),
    )
    nested = SafeObjectSlot(
        slot_id="nested",
        blueprint=SafeObjectConstructionBlueprint(
            module_identity=module_identity,
            class_name="Parcel",
            constructor_arguments=(("label", "piece"),),
        ),
        relation_depth=1,
        owner_slot_id="root",
    )
    return SafeObjectSetupPlan(
        module_identity=module_identity,
        target_identity="inspect",
        object_slots=(root, nested),
        setup_calls=(
            SafeSetupCall(
                receiver_slot_id="root",
                method_summary=summary,
                arguments=(
                    SafeTypedArgumentBinding(
                        parameter_name="amount",
                        type_name="int",
                        value=1,
                    ),
                ),
            ),
            SafeSetupCall(
                receiver_slot_id="root",
                method_summary=summary,
                arguments=(
                    SafeTypedArgumentBinding(
                        parameter_name="amount",
                        type_name="int",
                        value=2,
                    ),
                ),
            ),
        ),
        target_bindings=(
            SafeTargetBinding(
                parameter_name="subject",
                object_slot_id="root",
            ),
            SafeTargetBinding(
                parameter_name="item",
                object_slot_id="nested",
            ),
        ),
        receiver_slot_id=receiver_slot_id,
    )


def _call_leaf_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_generate_constructs_root_and_nested_objects_then_replays_ordered_setup() -> None:
    plan = create_runtime_setup_plan()
    scenario = replace(
        create_scenario(
            keyword_arguments=(),
            expected_result=3,
        ),
        setup_plan=plan,
    )

    source_code = PytestGenerator().generate(
        module_path="example.runtime_setup",
        function_name="inspect",
        scenarios=[scenario],
    )
    tree = ast.parse(source_code)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    test_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    calls = sorted(
        (
            (node.lineno, node.col_offset, _call_leaf_name(node))
            for node in ast.walk(test_function)
            if isinstance(node, ast.Call)
        ),
    )
    runtime_order = [
        name
        for _, _, name in calls
        if name in {"Bucket", "Parcel", "add", "inspect"}
    ]
    target_call = next(
        node
        for node in ast.walk(test_function)
        if isinstance(node, ast.Call) and _call_leaf_name(node) == "inspect"
    )

    assert imported == {"Bucket", "Parcel", "inspect"}
    assert runtime_order == ["Bucket", "Parcel", "add", "add", "inspect"]
    assert tuple(keyword.arg for keyword in target_call.keywords) == (
        "subject",
        "item",
    )
    assert all(isinstance(keyword.value, ast.Name) for keyword in target_call.keywords)
    assert source_code.index("amount=1") < source_code.index("amount=2")
    assert source_code.index("amount=2") < source_code.index("result = inspect(")
    assert "setattr(" not in source_code
    assert ".__dict__" not in source_code
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(
            isinstance(target, ast.Attribute)
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else (node.target,)
            )
        )
        for node in ast.walk(test_function)
    )


def test_generate_keeps_setup_calls_outside_target_expected_exception_scope() -> None:
    plan = create_runtime_setup_plan()
    scenario = replace(
        create_scenario(
            keyword_arguments=(),
            expected_result=None,
            expected_exception="LookupError",
        ),
        setup_plan=plan,
    )

    source_code = PytestGenerator().generate(
        module_path="example.runtime_setup",
        function_name="inspect",
        scenarios=[scenario],
    )
    tree = ast.parse(source_code)
    test_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    exception_scope = next(
        node for node in test_function.body if isinstance(node, ast.With)
    )
    setup_calls = [
        node
        for node in ast.walk(test_function)
        if isinstance(node, ast.Call) and _call_leaf_name(node) == "add"
    ]
    scoped_calls = {
        _call_leaf_name(node)
        for node in ast.walk(exception_scope)
        if isinstance(node, ast.Call)
    }

    assert len(setup_calls) == 2
    assert all(node.lineno < exception_scope.lineno for node in setup_calls)
    assert "inspect" in scoped_calls
    assert "add" not in scoped_calls


def test_project_generator_replays_setup_plan_before_target(
    tmp_path: Path,
) -> None:
    plan = create_runtime_setup_plan()
    scenario = replace(
        create_scenario(keyword_arguments=(), expected_result=3),
        setup_plan=plan,
    )
    source_file = tmp_path / "runtime_setup.py"
    source_file.write_text("def inspect(subject, item):\n    return 3\n", encoding="utf-8")
    candidate = ProjectTestCandidate(
        project_test_id="runtime::inspect::001",
        relative_module_path="runtime_setup.py",
        module_path="example.runtime_setup",
        function_name="inspect",
        function_start_line=1,
        function_end_line=2,
        source_file=source_file,
        import_root=tmp_path,
        scenario=scenario,
        original_order=1,
    )

    generated = ProjectPytestGenerator().generate((candidate,))

    assert "Bucket as _project_object_class_0001_001" in generated
    assert "Parcel as _project_object_class_0001_002" in generated
    assert "setup_object_0001_001 = _project_object_class_0001_001()" in generated
    assert "setup_object_0001_002 = _project_object_class_0001_002(label='piece')" in generated
    assert generated.index("amount=1") < generated.index("amount=2")
    assert generated.index("amount=2") < generated.index("_project_target_0001(")
    assert "subject=setup_object_0001_001" in generated
    assert "item=setup_object_0001_002" in generated
    assert "setattr(" not in generated
    assert ".__dict__" not in generated


def test_generate_constructs_instance_and_calls_method() -> None:
    generator = PytestGenerator()
    scenario = create_scenario(
        keyword_arguments=(("delta", 2),),
        expected_result="high",
    )
    scenario = replace(
        scenario,
        constructor_arguments=(("value", 8),),
        target_class_name="Counter",
    )

    source_code = generator.generate(
        module_path="example.counter",
        function_name="classify",
        scenarios=[scenario],
    )

    assert "from example.counter import Counter" in source_code
    assert "target = Counter(value=8)" in source_code
    assert "result = target.classify(delta=2)" in source_code
    assert "assert result == 'high'" in source_code
    assert "self=" not in source_code
    ast.parse(source_code)


def test_generate_uses_setup_root_as_method_receiver_without_duplicate_constructor() -> None:
    plan = create_runtime_setup_plan(receiver_slot_id="root")
    scenario = replace(
        create_scenario(keyword_arguments=(), expected_result=3),
        setup_plan=plan,
        constructor_arguments=(),
        target_class_name="Bucket",
    )

    source_code = PytestGenerator().generate(
        module_path="example.runtime_setup",
        function_name="inspect",
        scenarios=[scenario],
    )

    assert source_code.count("Bucket(") == 1
    assert "target = setup_object_root" in source_code
    assert "result = target.inspect(" in source_code
    assert "self=" not in source_code


def test_project_generator_uses_setup_root_as_method_receiver_without_duplicate_constructor(
    tmp_path: Path,
) -> None:
    plan = create_runtime_setup_plan(receiver_slot_id="root")
    scenario = replace(
        create_scenario(keyword_arguments=(), expected_result=3),
        setup_plan=plan,
        constructor_arguments=(),
        target_class_name="Bucket",
    )
    source_file = tmp_path / "runtime_setup.py"
    source_file.write_text(
        "class Bucket:\n    def inspect(self):\n        return 3\n",
        encoding="utf-8",
    )
    candidate = ProjectTestCandidate(
        project_test_id="runtime::Bucket.inspect::001",
        relative_module_path="runtime_setup.py",
        module_path="example.runtime_setup",
        function_name="inspect",
        function_start_line=1,
        function_end_line=3,
        source_file=source_file,
        import_root=tmp_path,
        scenario=scenario,
        original_order=1,
    )

    generated = ProjectPytestGenerator().generate((candidate,))

    assert generated.count("_project_object_class_0001_001()") == 1
    assert "target = setup_object_0001_001" in generated
    assert "result = target.inspect(" in generated
    assert "self=" not in generated


def test_generate_creates_expected_result_assertion() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(
                expected_result="Başarılı",
            ),
        ],
    )

    assert (
        "assert result == 'Başarılı'"
        in source_code
    )


def test_generate_does_not_create_callable_placeholder() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(),
        ],
    )

    assert "assert callable" not in source_code
    assert "_target_function" not in source_code
    assert "TODO" not in source_code


def test_generate_creates_one_test_for_each_scenario() -> None:
    generator = PytestGenerator()

    scenarios = [
        create_scenario(
            scenario_id="scenario_001",
            path_index=1,
            priority_rank=1,
            keyword_arguments=(("score", 85),),
            expected_result="Başarılı",
        ),
        create_scenario(
            scenario_id="scenario_002",
            path_index=2,
            priority_rank=2,
            keyword_arguments=(("score", 50),),
            expected_result="Orta",
        ),
        create_scenario(
            scenario_id="scenario_003",
            path_index=3,
            priority_rank=3,
            keyword_arguments=(("score", 49),),
            expected_result="Başarısız",
        ),
    ]

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=scenarios,
    )

    assert source_code.count(
        "def test_calculate_score_"
    ) == 3

    assert (
        "calculate_score(score=85)"
        in source_code
    )
    assert (
        "calculate_score(score=50)"
        in source_code
    )
    assert (
        "calculate_score(score=49)"
        in source_code
    )


def test_generate_creates_exception_test() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(
                keyword_arguments=(
                    ("score", -1),
                ),
                expected_result=None,
                expected_exception="ValueError",
            ),
        ],
    )

    assert "import pytest" in source_code
    assert (
        "with pytest.raises(ValueError):"
        in source_code
    )
    assert (
        "calculate_score(score=-1)"
        in source_code
    )


def test_generate_does_not_import_pytest_without_exception() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(),
        ],
    )

    assert "import pytest" not in source_code


def test_generate_preserves_safe_scenario_metadata_without_dqm_score() -> None:
    generator = PytestGenerator()

    source_code = generator.generate(
        module_path="datasets.sample_code",
        function_name="calculate_score",
        scenarios=[
            create_scenario(
                scenario_id=(
                    "calculate_score_scenario_001"
                ),
                priority_level="High",
                dqm_score=100.0,
            ),
        ],
    )

    assert (
        "# Senaryo kimliği: "
        "calculate_score_scenario_001"
    ) in source_code

    assert (
        "# DQM önceliği: High"
    ) in source_code

    assert "Normalize DQM skoru" not in source_code
    assert "100.0" not in source_code


def test_generate_omits_raw_dqm_scores_without_changing_scenario_order() -> None:
    source_code = PytestGenerator().generate(
        module_path="example.metrics",
        function_name="classify",
        scenarios=(
            create_scenario(
                scenario_id="first-safe-identity",
                path_index=7,
                priority_rank=1,
                priority_level="High",
                dqm_score=98.7654321,
            ),
            create_scenario(
                scenario_id="second-safe-identity",
                path_index=3,
                priority_rank=2,
                priority_level="Low",
                dqm_score=12.3456789,
            ),
        ),
    )

    assert source_code.index("first-safe-identity") < source_code.index(
        "second-safe-identity"
    )
    assert "98.7654321" not in source_code
    assert "12.3456789" not in source_code
    assert "Normalize DQM skoru" not in source_code


def test_project_generator_omits_raw_dqm_scores_and_preserves_candidate_order(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "metrics.py"
    source_file.write_text(
        "def classify(score):\n    return score\n",
        encoding="utf-8",
    )
    candidates = tuple(
        ProjectTestCandidate(
            project_test_id=project_test_id,
            relative_module_path="metrics.py",
            module_path="example.metrics",
            function_name="classify",
            function_start_line=1,
            function_end_line=2,
            source_file=source_file,
            import_root=tmp_path,
            scenario=create_scenario(
                scenario_id=scenario_id,
                path_index=path_index,
                priority_rank=priority_rank,
                dqm_score=dqm_score,
            ),
            original_order=priority_rank,
        )
        for (
            project_test_id,
            scenario_id,
            path_index,
            priority_rank,
            dqm_score,
        ) in (
            ("candidate-high", "scenario-high", 9, 1, 91.2345678),
            ("candidate-low", "scenario-low", 2, 2, 10.8765432),
        )
    )

    source_code = ProjectPytestGenerator().generate(candidates)

    assert source_code.index("candidate-high") < source_code.index(
        "candidate-low"
    )
    assert "91.2345678" not in source_code
    assert "10.8765432" not in source_code
    assert "Normalize DQM skoru" not in source_code


def test_generate_rejects_empty_module_path() -> None:
    generator = PytestGenerator()

    with pytest.raises(
        ValueError,
        match="Modül yolu boş olamaz",
    ):
        generator.generate(
            module_path=" ",
            function_name="calculate_score",
            scenarios=[
                create_scenario(),
            ],
        )


def test_generate_rejects_invalid_module_path() -> None:
    generator = PytestGenerator()

    with pytest.raises(
        ValueError,
        match="Geçersiz Python modül yolu",
    ):
        generator.generate(
            module_path="datasets/sample_code",
            function_name="calculate_score",
            scenarios=[
                create_scenario(),
            ],
        )


def test_generate_rejects_empty_function_name() -> None:
    generator = PytestGenerator()

    with pytest.raises(
        ValueError,
        match="Fonksiyon adı boş olamaz",
    ):
        generator.generate(
            module_path="datasets.sample_code",
            function_name=" ",
            scenarios=[
                create_scenario(),
            ],
        )


def test_generate_rejects_empty_scenarios() -> None:
    generator = PytestGenerator()

    with pytest.raises(
        ValueError,
        match="En az bir test senaryosu gereklidir",
    ):
        generator.generate(
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=[],
        )


def test_generate_rejects_invalid_scenario_type() -> None:
    generator = PytestGenerator()

    with pytest.raises(
        TypeError,
        match="Bütün senaryolar Scenario türünde olmalıdır",
    ):
        generator.generate(
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=[
                "invalid",  # type: ignore[list-item]
            ],
        )


def test_generate_rejects_duplicate_test_names() -> None:
    generator = PytestGenerator()

    duplicate_scenarios = [
        create_scenario(
            scenario_id="scenario_001",
            path_index=1,
            priority_rank=1,
        ),
        create_scenario(
            scenario_id="scenario_002",
            path_index=1,
            priority_rank=1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Aynı pytest fonksiyon adı",
    ):
        generator.generate(
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=duplicate_scenarios,
        )


def test_generate_rejects_duplicate_keyword_arguments() -> None:
    generator = PytestGenerator()

    scenario = create_scenario(
        keyword_arguments=(
            ("score", 85),
            ("score", 90),
        ),
    )

    with pytest.raises(
        ValueError,
        match="tekrar eden fonksiyon parametreleri",
    ):
        generator.generate(
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=[scenario],
        )


def test_generate_rejects_invalid_expected_exception_name() -> None:
    generator = PytestGenerator()

    scenario = create_scenario(
        expected_result=None,
        expected_exception="Value Error",
    )

    with pytest.raises(
        ValueError,
        match="Beklenen exception adı geçersizdir",
    ):
        generator.generate(
            module_path="datasets.sample_code",
            function_name="calculate_score",
            scenarios=[scenario],
        )
