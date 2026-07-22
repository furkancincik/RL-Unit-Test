import ast
from collections.abc import Sequence
from typing import Final

import pytest

from analyzer.python_analyzer import PythonAnalyzer
from cfg.control_flow_graph import ControlFlowGraphBuilder
from cfg.path_analyzer import CFGPathAnalyzer
from evaluator.dqm import DecisionQualityMatrix
from generator.pytest_generator import PytestGenerator
from generator.scenario_generator import (
    TestScenario as ScenarioModel,
    TestScenarioGenerator,
)


SOURCE_FILE: Final[str] = "datasets/sample_code.py"
MODULE_PATH: Final[str] = "datasets.sample_code"
FUNCTION_NAME: Final[str] = "calculate_score"


def create_generated_pytest_code() -> tuple[str, str, int]:
    """
    GerÃ§ek analiz iÅŸlem hattÄ±nÄ± kullanarak pytest kaynak kodu Ã¼retir.

    Returns:
        Ãœretilen pytest kaynak kodunu, analiz edilen fonksiyonun adÄ±nÄ±
        ve oluÅŸturulan senaryo sayÄ±sÄ±nÄ± iÃ§eren tuple.
    """
    analyzer = PythonAnalyzer()
    cfg_builder = ControlFlowGraphBuilder()
    path_analyzer = CFGPathAnalyzer()
    dqm = DecisionQualityMatrix()
    scenario_generator = TestScenarioGenerator()
    pytest_generator = PytestGenerator()

    analysis_result = analyzer.analyze_file(SOURCE_FILE)
    graphs = cfg_builder.build_from_file(SOURCE_FILE)

    assert analysis_result.functions
    assert graphs

    function = analysis_result.functions[0]
    graph = graphs[0]

    paths = path_analyzer.find_paths(graph)

    scores = dqm.evaluate_paths(
        function=function,
        paths=paths,
    )

    scenarios = scenario_generator.generate(
        function_name=function.name,
        paths=paths,
        scores=scores,
    )

    generated_code = pytest_generator.generate(
        module_path=MODULE_PATH,
        function_name=function.name,
        scenarios=scenarios,
    )

    return generated_code, function.name, len(scenarios)


def create_minimal_scenario(
    *,
    scenario_id: str = "sample_scenario_001",
    path_index: int = 1,
    priority_rank: int = 1,
    description: str = "Ã–rnek test senaryosu.",
) -> ScenarioModel:
    """
    PytestGenerator testlerinde kullanÄ±lacak minimal senaryo oluÅŸturur.

    Args:
        scenario_id: Senaryonun benzersiz kimliÄŸi.
        path_index: Ä°liÅŸkili yÃ¼rÃ¼tme yolunun numarasÄ±.
        priority_rank: DQM Ã¶ncelik sÄ±rasÄ±.
        description: Senaryo aÃ§Ä±klamasÄ±.

    Returns:
        Testlerde kullanÄ±lmaya hazÄ±r TestScenario nesnesi.
    """
    return ScenarioModel(
        scenario_id=scenario_id,
        name="Ã–rnek senaryo",
        path_index=path_index,
        priority_rank=priority_rank,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1, 2),
        edge_labels=("true",),
        contains_loop=False,
        contains_exception=False,
        description=description,
    )


def test_generate_creates_valid_python_code() -> None:
    """Ãœretilen metnin geÃ§erli Python kaynak kodu olduÄŸunu doÄŸrular."""
    generated_code, _, _ = create_generated_pytest_code()

    syntax_tree = ast.parse(generated_code)

    assert isinstance(syntax_tree, ast.Module)


def test_generate_creates_test_for_each_scenario() -> None:
    """Her senaryo iÃ§in ayrÄ± bir pytest fonksiyonu Ã¼retildiÄŸini doÄŸrular."""
    generated_code, function_name, scenario_count = (
        create_generated_pytest_code()
    )

    generated_test_count = generated_code.count(
        f"def test_{function_name}_"
    )

    assert scenario_count == 3
    assert generated_test_count == scenario_count


def test_generate_contains_target_import() -> None:
    """Ãœretilen kaynak kodda hedef fonksiyon importunu doÄŸrular."""
    generated_code, function_name, _ = (
        create_generated_pytest_code()
    )

    expected_import = (
        f"from {MODULE_PATH} import {function_name}"
    )

    assert expected_import in generated_code


def test_generate_contains_scenario_metadata() -> None:
    """Ãœretilen testlerde senaryo metadata bilgilerinin yer aldÄ±ÄŸÄ±nÄ± doÄŸrular."""
    generated_code, _, _ = create_generated_pytest_code()

    assert "Senaryo kimliÄŸi:" in generated_code
    assert "DQM Ã¶nceliÄŸi:" in generated_code
    assert "Normalize DQM skoru:" in generated_code
    assert "CFG dÃ¼ÄŸÃ¼m yolu:" in generated_code
    assert "CFG kenarlarÄ±:" in generated_code


def test_generate_contains_callable_assertion() -> None:
    """Ãœretilen testlerin hedef fonksiyonun Ã§aÄŸrÄ±labilirliÄŸini doÄŸruladÄ±ÄŸÄ±nÄ± test eder."""
    generated_code, _, _ = create_generated_pytest_code()

    assert "_target_function = calculate_score" in generated_code
    assert "assert callable(_target_function)" in generated_code


def test_generate_does_not_include_unnecessary_pass_statement() -> None:
    """Ãœretilen test gÃ¶vdelerinde gereksiz pass ifadesi bulunmadÄ±ÄŸÄ±nÄ± doÄŸrular."""
    generated_code, _, _ = create_generated_pytest_code()

    assert "\n    pass\n" not in generated_code
    assert (
        "# TODO: Senaryoya uygun girdiler ve doÄŸrulamalar Ã¼retilecek."
        in generated_code
    )


def test_generate_safely_handles_special_characters_in_description() -> None:
    """Ã–zel karakter iÃ§eren aÃ§Ä±klamalarÄ±n Python sÃ¶zdizimini bozmadÄ±ÄŸÄ±nÄ± doÄŸrular."""
    generator = PytestGenerator()

    scenario = create_minimal_scenario(
        description=(
            'AÃ§Ä±klama "Ã§ift tÄ±rnak", \'tek tÄ±rnak\' ve\n'
            'yeni satÄ±r iÃ§eriyor.'
        ),
    )

    generated_code = generator.generate(
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=[scenario],
    )

    syntax_tree = ast.parse(generated_code)

    assert isinstance(syntax_tree, ast.Module)
    assert "AÃ§Ä±klama" in generated_code


@pytest.mark.parametrize(
    ("module_path", "function_name", "expected_message"),
    [
        (
            " ",
            FUNCTION_NAME,
            "ModÃ¼l yolu boÅŸ olamaz",
        ),
        (
            MODULE_PATH,
            " ",
            "Fonksiyon adÄ± boÅŸ olamaz",
        ),
        (
            "datasets/sample_code",
            FUNCTION_NAME,
            "GeÃ§ersiz Python modÃ¼l yolu",
        ),
        (
            "datasets..sample_code",
            FUNCTION_NAME,
            "GeÃ§ersiz Python modÃ¼l yolu",
        ),
        (
            "datasets.class",
            FUNCTION_NAME,
            "GeÃ§ersiz Python modÃ¼l yolu",
        ),
        (
            MODULE_PATH,
            "calculate-score",
            "GeÃ§ersiz Python fonksiyon adÄ±",
        ),
        (
            MODULE_PATH,
            "class",
            "GeÃ§ersiz Python fonksiyon adÄ±",
        ),
    ],
)
def test_generate_rejects_invalid_names(
    module_path: str,
    function_name: str,
    expected_message: str,
) -> None:
    """GeÃ§ersiz modÃ¼l yollarÄ± ve fonksiyon adlarÄ±nÄ±n reddedildiÄŸini doÄŸrular."""
    generator = PytestGenerator()

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        generator.generate(
            module_path=module_path,
            function_name=function_name,
            scenarios=[
                create_minimal_scenario(),
            ],
        )


def test_generate_rejects_empty_scenario_list() -> None:
    """BoÅŸ senaryo koleksiyonuyla test Ã¼retilemediÄŸini doÄŸrular."""
    generator = PytestGenerator()

    with pytest.raises(
        ValueError,
        match="En az bir test senaryosu gereklidir",
    ):
        generator.generate(
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            scenarios=[],
        )


def test_generate_rejects_invalid_scenario_type() -> None:
    """TestScenario dÄ±ÅŸÄ±ndaki nesnelerin reddedildiÄŸini doÄŸrular."""
    generator = PytestGenerator()

    invalid_scenarios: Sequence[object] = [
        object(),
    ]

    with pytest.raises(
        TypeError,
        match="BÃ¼tÃ¼n senaryolar TestScenario tÃ¼rÃ¼nde olmalÄ±dÄ±r",
    ):
        generator.generate(
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            scenarios=invalid_scenarios,  # type: ignore[arg-type]
        )


def test_generate_rejects_empty_scenario_id() -> None:
    """BoÅŸ senaryo kimliÄŸinin reddedildiÄŸini doÄŸrular."""
    generator = PytestGenerator()

    scenario = create_minimal_scenario(
        scenario_id=" ",
    )

    with pytest.raises(
        ValueError,
        match="Senaryo kimliÄŸi boÅŸ olamaz",
    ):
        generator.generate(
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            scenarios=[scenario],
        )


@pytest.mark.parametrize(
    ("path_index", "priority_rank", "expected_message"),
    [
        (
            0,
            1,
            "YÃ¼rÃ¼tme yolu numarasÄ± 1 veya daha bÃ¼yÃ¼k olmalÄ±dÄ±r",
        ),
        (
            -1,
            1,
            "YÃ¼rÃ¼tme yolu numarasÄ± 1 veya daha bÃ¼yÃ¼k olmalÄ±dÄ±r",
        ),
        (
            1,
            0,
            "Ã–ncelik sÄ±rasÄ± 1 veya daha bÃ¼yÃ¼k olmalÄ±dÄ±r",
        ),
        (
            1,
            -1,
            "Ã–ncelik sÄ±rasÄ± 1 veya daha bÃ¼yÃ¼k olmalÄ±dÄ±r",
        ),
    ],
)
def test_generate_rejects_invalid_scenario_indices(
    path_index: int,
    priority_rank: int,
    expected_message: str,
) -> None:
    """GeÃ§ersiz yol ve Ã¶ncelik numaralarÄ±nÄ±n reddedildiÄŸini doÄŸrular."""
    generator = PytestGenerator()

    scenario = create_minimal_scenario(
        path_index=path_index,
        priority_rank=priority_rank,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        generator.generate(
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            scenarios=[scenario],
        )


def test_generate_rejects_duplicate_test_function_names() -> None:
    """AynÄ± isimde iki pytest fonksiyonu Ã¼retilmesini engeller."""
    generator = PytestGenerator()

    first_scenario = create_minimal_scenario(
        scenario_id="scenario_001",
        path_index=1,
        priority_rank=1,
    )
    second_scenario = create_minimal_scenario(
        scenario_id="scenario_002",
        path_index=1,
        priority_rank=1,
    )

    with pytest.raises(
        ValueError,
        match="AynÄ± pytest fonksiyon adÄ± birden fazla kez Ã¼retilemez",
    ):
        generator.generate(
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            scenarios=[
                first_scenario,
                second_scenario,
            ],
        )


def test_generate_creates_stable_test_function_name() -> None:
    """Yol ve Ã¶ncelik deÄŸerlerinden kararlÄ± test adÄ± Ã¼retildiÄŸini doÄŸrular."""
    generator = PytestGenerator()

    scenario = create_minimal_scenario(
        path_index=2,
        priority_rank=3,
    )

    generated_code = generator.generate(
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=[scenario],
    )

    assert (
        "def test_calculate_score_path_2_priority_3() -> None:"
        in generated_code
    )


def test_generate_ends_with_single_newline() -> None:
    """Ãœretilen kaynak kodun tek satÄ±r sonu karakteriyle bittiÄŸini doÄŸrular."""
    generator = PytestGenerator()

    generated_code = generator.generate(
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=[
            create_minimal_scenario(),
        ],
    )

    assert generated_code.endswith("\n")
    assert not generated_code.endswith("\n\n")

