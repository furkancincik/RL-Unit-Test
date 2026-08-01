from __future__ import annotations

import ast

import pytest

from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import Scenario


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


def test_generate_preserves_scenario_metadata() -> None:
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

    assert (
        "# Normalize DQM skoru: 100.0"
    ) in source_code


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