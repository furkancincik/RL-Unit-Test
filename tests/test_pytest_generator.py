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
    Gerçek analiz işlem hattını kullanarak pytest kaynak kodu üretir.

    Returns:
        Üretilen pytest kaynak kodunu, analiz edilen fonksiyonun adını
        ve oluşturulan senaryo sayısını içeren tuple.
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
    description: str = "Örnek test senaryosu.",
) -> ScenarioModel:
    """
    PytestGenerator testlerinde kullanılacak minimal senaryo oluşturur.

    Args:
        scenario_id: Senaryonun benzersiz kimliği.
        path_index: İlişkili yürütme yolunun numarası.
        priority_rank: DQM öncelik sırası.
        description: Senaryo açıklaması.

    Returns:
        Testlerde kullanılmaya hazır TestScenario nesnesi.
    """
    return ScenarioModel(
        scenario_id=scenario_id,
        name="Örnek senaryo",
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
    """Üretilen metnin geçerli Python kaynak kodu olduğunu doğrular."""
    generated_code, _, _ = create_generated_pytest_code()

    syntax_tree = ast.parse(generated_code)

    assert isinstance(syntax_tree, ast.Module)


def test_generate_creates_test_for_each_scenario() -> None:
    """Her senaryo için ayrı bir pytest fonksiyonu üretildiğini doğrular."""
    generated_code, function_name, scenario_count = (
        create_generated_pytest_code()
    )

    generated_test_count = generated_code.count(
        f"def test_{function_name}_"
    )

    assert scenario_count == 3
    assert generated_test_count == scenario_count


def test_generate_contains_target_import() -> None:
    """Üretilen kaynak kodda hedef fonksiyon importunu doğrular."""
    generated_code, function_name, _ = (
        create_generated_pytest_code()
    )

    expected_import = (
        f"from {MODULE_PATH} import {function_name}"
    )

    assert expected_import in generated_code


def test_generate_contains_scenario_metadata() -> None:
    """Üretilen testlerde senaryo metadata bilgilerinin yer aldığını doğrular."""
    generated_code, _, _ = create_generated_pytest_code()

    assert "Senaryo kimliği:" in generated_code
    assert "DQM önceliği:" in generated_code
    assert "Normalize DQM skoru:" in generated_code
    assert "CFG düğüm yolu:" in generated_code
    assert "CFG kenarları:" in generated_code


def test_generate_contains_callable_assertion() -> None:
    """Üretilen testlerin hedef fonksiyonun çağrılabilirliğini doğruladığını test eder."""
    generated_code, _, _ = create_generated_pytest_code()

    assert "_target_function = calculate_score" in generated_code
    assert "assert callable(_target_function)" in generated_code


def test_generate_does_not_include_unnecessary_pass_statement() -> None:
    """Üretilen test gövdelerinde gereksiz pass ifadesi bulunmadığını doğrular."""
    generated_code, _, _ = create_generated_pytest_code()

    assert "\n    pass\n" not in generated_code
    assert (
        "# TODO: Senaryoya uygun girdiler ve doğrulamalar üretilecek."
        in generated_code
    )


def test_generate_safely_handles_special_characters_in_description() -> None:
    """Özel karakter içeren açıklamaların Python sözdizimini bozmadığını doğrular."""
    generator = PytestGenerator()

    scenario = create_minimal_scenario(
        description=(
            'Açıklama "çift tırnak", \'tek tırnak\' ve\n'
            'yeni satır içeriyor.'
        ),
    )

    generated_code = generator.generate(
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        scenarios=[scenario],
    )

    syntax_tree = ast.parse(generated_code)

    assert isinstance(syntax_tree, ast.Module)
    assert "Açıklama" in generated_code


@pytest.mark.parametrize(
    ("module_path", "function_name", "expected_message"),
    [
        (
            " ",
            FUNCTION_NAME,
            "Modül yolu boş olamaz",
        ),
        (
            MODULE_PATH,
            " ",
            "Fonksiyon adı boş olamaz",
        ),
        (
            "datasets/sample_code",
            FUNCTION_NAME,
            "Geçersiz Python modül yolu",
        ),
        (
            "datasets..sample_code",
            FUNCTION_NAME,
            "Geçersiz Python modül yolu",
        ),
        (
            "datasets.class",
            FUNCTION_NAME,
            "Geçersiz Python modül yolu",
        ),
        (
            MODULE_PATH,
            "calculate-score",
            "Geçersiz Python fonksiyon adı",
        ),
        (
            MODULE_PATH,
            "class",
            "Geçersiz Python fonksiyon adı",
        ),
    ],
)
def test_generate_rejects_invalid_names(
    module_path: str,
    function_name: str,
    expected_message: str,
) -> None:
    """Geçersiz modül yolları ve fonksiyon adlarının reddedildiğini doğrular."""
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
    """Boş senaryo koleksiyonuyla test üretilemediğini doğrular."""
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
    """TestScenario dışındaki nesnelerin reddedildiğini doğrular."""
    generator = PytestGenerator()

    invalid_scenarios: Sequence[object] = [
        object(),
    ]

    with pytest.raises(
        TypeError,
        match="Bütün senaryolar TestScenario türünde olmalıdır",
    ):
        generator.generate(
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            scenarios=invalid_scenarios,  # type: ignore[arg-type]
        )


def test_generate_rejects_empty_scenario_id() -> None:
    """Boş senaryo kimliğinin reddedildiğini doğrular."""
    generator = PytestGenerator()

    scenario = create_minimal_scenario(
        scenario_id=" ",
    )

    with pytest.raises(
        ValueError,
        match="Senaryo kimliği boş olamaz",
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
            "Yürütme yolu numarası 1 veya daha büyük olmalıdır",
        ),
        (
            -1,
            1,
            "Yürütme yolu numarası 1 veya daha büyük olmalıdır",
        ),
        (
            1,
            0,
            "Öncelik sırası 1 veya daha büyük olmalıdır",
        ),
        (
            1,
            -1,
            "Öncelik sırası 1 veya daha büyük olmalıdır",
        ),
    ],
)
def test_generate_rejects_invalid_scenario_indices(
    path_index: int,
    priority_rank: int,
    expected_message: str,
) -> None:
    """Geçersiz yol ve öncelik numaralarının reddedildiğini doğrular."""
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
    """Aynı isimde iki pytest fonksiyonu üretilmesini engeller."""
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
        match="Aynı pytest fonksiyon adı birden fazla kez üretilemez",
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
    """Yol ve öncelik değerlerinden kararlı test adı üretildiğini doğrular."""
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
    """Üretilen kaynak kodun tek satır sonu karakteriyle bittiğini doğrular."""
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