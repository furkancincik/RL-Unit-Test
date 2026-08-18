from __future__ import annotations

from pathlib import Path

import pytest

from models.coverage_result import FunctionCoverageResult
from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineRunStatus,
    PipelineStage,
)
from services.pipeline_timeout_service import (
    PipelineDiagnosticCheckpointStore,
)
from services.real_rl_training_service import (
    RealRLTrainingResult,
    RealRLTrainingService,
)


SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"
FUNCTION_NAME = "calculate_score"


def test_real_rl_training_service_runs_end_to_end(
    tmp_path: Path,
) -> None:
    """
    Kaynak kod analizinden gerçek coverage tabanlı RL eğitimine
    kadar bütün hattın birlikte çalıştığını doğrular.

    Bu entegrasyon testi deterministik davranışı korumak için
    epsilon=0.0 ve epsilon_decay_rate=None kullanır.
    """
    service = RealRLTrainingService()

    result = service.run(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path,
        episode_count=3,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        random_seed=42,
        overwrite=True,
        timeout_seconds=30.0,
    )

    assert isinstance(
        result,
        RealRLTrainingResult,
    )

    assert result.success is True
    assert result.source_file == SOURCE_FILE.resolve()
    assert result.module_path == MODULE_PATH
    assert result.function_name == FUNCTION_NAME

    assert result.scenario_count >= 1
    assert result.completed_episode_count == 3

    assert result.statistics.episode_count == 3
    assert result.statistics.total_reward > 0.0
    assert result.statistics.average_reward > 0.0

    assert (
        result.statistics.best_coverage_percentage
        == 100.0
    )

    assert (
        result.statistics.full_coverage_episode_count
        == 3
    )

    assert result.q_table_state_count > 0
    assert result.session_result.best_episode is not None

    assert all(
        episode.full_coverage
        for episode in result.session_result.episodes
    )

    assert all(
        episode.final_coverage_percentage == 100.0
        for episode in result.session_result.episodes
    )

    assert isinstance(
        result.final_coverage_result,
        FunctionCoverageResult,
    )
    assert result.function_coverage is not None
    assert result.has_full_function_coverage is True
    assert result.has_full_file_coverage is True
    assert result.file_coverage.success is True
    assert result.diagnostic is not None
    assert result.diagnostic.status is PipelineRunStatus.COMPLETED
    assert result.diagnostic.funnel.final_scenario_count == result.scenario_count
    assert result.diagnostic.line_coverage_percent == 100.0


def test_real_rl_training_service_generates_cumulative_test_file(
    tmp_path: Path,
) -> None:
    """
    Gerçek RL eğitimi sırasında kümülatif pytest dosyasının
    oluşturulduğunu ve çalıştırılabilir testler içerdiğini doğrular.
    """
    result = RealRLTrainingService().run(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path,
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        random_seed=42,
        overwrite=True,
        timeout_seconds=30.0,
    )

    generated_test_file = (
        tmp_path
        / "test_calculate_score_scenario_suite.py"
    )

    assert result.success is True
    assert generated_test_file.exists()

    generated_source = generated_test_file.read_text(
        encoding="utf-8",
    )

    assert (
        "from datasets.sample_code import calculate_score"
        in generated_source
    )

    assert "def test_calculate_score_" in generated_source
    assert "result = calculate_score(" in generated_source
    assert "assert result ==" in generated_source

    assert "assert callable" not in generated_source
    assert "TODO" not in generated_source


def test_real_rl_training_service_produces_readable_report(
    tmp_path: Path,
) -> None:
    """
    Eğitim oturumu sonunda fonksiyon ve dosya coverage değerlerini
    ayıran okunabilir terminal raporunun üretildiğini doğrular.
    """
    result = RealRLTrainingService().run(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path,
        episode_count=2,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        random_seed=42,
        overwrite=True,
        timeout_seconds=30.0,
    )

    report = result.report

    assert isinstance(report, str)
    assert report.strip()

    assert "RL EĞİTİM OTURUMU" in report
    assert "Episode 1" in report
    assert "Episode 2" in report
    assert "GENEL ÖZET" in report

    assert "Hedef kapsam coverage" in report
    assert "SON COVERAGE ÖZETİ" in report
    assert "Hedef fonksiyon" in report
    assert "Fonksiyon satır coverage" in report
    assert "Fonksiyon branch coverage" in report
    assert "Tam fonksiyon coverage" in report
    assert "Dosya geneli satır coverage" in report
    assert "Dosya geneli branch coverage" in report
    assert "Tam dosya coverage" in report

    assert "Toplam reward" in report
    assert "En iyi episode" in report
    assert "Oturum başarılı" in report


def test_real_pipeline_returns_partial_diagnostic_for_controlled_failure(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "unsupported_result.py"
    source_file.write_text(
        "def helper(value):\n"
        "    return value\n\n"
        "def unsupported_result(value: int):\n"
        "    return helper(value)\n",
        encoding="utf-8",
    )

    result = RealRLTrainingService().run_with_diagnostics(
        source_file=source_file,
        module_path="unsupported_result",
        function_name="unsupported_result",
        output_directory=tmp_path / "output",
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        timeout_seconds=30.0,
    )

    assert isinstance(result, PipelineDiagnosticResult)
    assert result.status is PipelineRunStatus.PARTIAL
    assert result.stopped_stage is PipelineStage.SCENARIO_GENERATION
    assert result.funnel.bounded_path_count is not None
    assert result.funnel.scenario_generation_accepted_count == 0
    assert dict(result.scenario_rejection_counts) == {
        "UNSUPPORTED_EXPECTED_RESULT": 1,
    }


def test_small_real_pipeline_completes_before_global_timeout(
    tmp_path: Path,
) -> None:
    result = RealRLTrainingService().run(
        source_file=Path("datasets/sample_robustness_code.py"),
        module_path="datasets.sample_robustness_code",
        function_name="calculate_category_usage",
        output_directory=tmp_path,
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        random_seed=42,
        timeout_seconds=30.0,
        pipeline_timeout_seconds=20.0,
    )

    assert isinstance(result, RealRLTrainingResult)
    assert result.scenario_count == 3
    assert result.diagnostic is not None
    assert result.diagnostic.status is PipelineRunStatus.COMPLETED
    assert result.diagnostic.pipeline_timeout_seconds == pytest.approx(20.0)
    assert result.diagnostic.funnel.final_scenario_count == 3
    assert result.diagnostic.line_coverage_percent == 100.0
    assert result.diagnostic.branch_coverage_percent == 100.0


def test_real_pipeline_timeout_returns_checkpoint_and_allows_next_run(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "slow_target.py"
    source_file.write_text(
        "import time\n\n"
        "def slow_target(value: int):\n"
        "    time.sleep(10)\n"
        "    return value\n",
        encoding="utf-8",
    )
    service = RealRLTrainingService()

    timed_out = service.run_with_diagnostics(
        source_file=source_file,
        module_path="slow_target",
        function_name="slow_target",
        output_directory=tmp_path / "slow-output",
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        pipeline_timeout_seconds=2.0,
    )

    assert isinstance(timed_out, PipelineDiagnosticResult)
    assert timed_out.status is PipelineRunStatus.TIMED_OUT
    assert timed_out.stopped_stage is PipelineStage.CONCRETE_VALIDATION
    assert timed_out.funnel.bounded_path_count == 1
    assert timed_out.funnel.pre_concrete_scenario_count == 1
    assert timed_out.line_coverage_percent is None
    assert timed_out.funnel.rl_executed_test_count is None

    completed = service.run(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path / "next-output",
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
        pipeline_timeout_seconds=20.0,
    )
    assert isinstance(completed, RealRLTrainingResult)
    assert completed.success is True


def test_completed_run_writes_matching_durable_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "diagnostic.json"
    service = RealRLTrainingService(
        diagnostic_checkpoint_path=checkpoint_path
    )
    result = service.run(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path / "output",
        episode_count=1,
        epsilon=0.0,
        epsilon_decay_rate=None,
        minimum_epsilon=0.0,
    )

    assert isinstance(result, RealRLTrainingResult)
    assert result.diagnostic is not None
    checkpoint = PipelineDiagnosticCheckpointStore(checkpoint_path).read()
    assert checkpoint is not None
    assert checkpoint.to_dict() == result.diagnostic.to_dict()
