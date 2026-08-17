from __future__ import annotations

from pathlib import Path

import pytest

from models.coverage_reachability_result import (
    FunctionCoverageReachabilityResult,
    LineReachabilityEvidence,
    LineReachabilityStatus,
)
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep
from rl.training_report_formatter import TrainingReportFormatter
from rl.training_session import TrainingSessionResult
from rl.training_statistics import TrainingStatistics


def create_training_results() -> tuple[
    TrainingSessionResult,
    TrainingStatistics,
]:
    """Aynı episode kaydını paylaşan oturum ve istatistik oluşturur."""
    steps = (
        EnvironmentStep(
            state=CoverageState(
                coverage_percentage=20.0,
                executed_tests=1,
                missing_lines=(3, 4, 5),
                uncovered_branches=2,
            ),
            reward=10.0,
            done=False,
        ),
        EnvironmentStep(
            state=CoverageState(
                coverage_percentage=50.0,
                executed_tests=2,
                missing_lines=(4, 5),
                uncovered_branches=1,
            ),
            reward=12.5,
            done=False,
        ),
        EnvironmentStep(
            state=CoverageState(
                coverage_percentage=80.0,
                executed_tests=3,
                missing_lines=(5,),
                uncovered_branches=1,
            ),
            reward=20.0,
            done=True,
        ),
    )

    statistics = TrainingStatistics()
    statistics.record_episode(
        steps=steps,
    )

    result = TrainingSessionResult(
        episodes=statistics.episodes,
        requested_episode_count=1,
        completed_episode_count=1,
    )

    return result, statistics


def create_function_coverage(
    *,
    source_file: Path = Path("datasets/sample.py"),
    function_name: str = "classify",
    line_coverage_percent: float = 40.0,
) -> FunctionCoverageResult:
    """Rapor testleri için fonksiyon ve dosya coverage sonucu üretir."""
    test_file = Path("output/test_classify.py")

    file_coverage = CoverageResult(
        source_file=source_file,
        test_file=test_file,
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=50.0,
        covered_line_count=2,
        missing_line_count=3,
        total_line_count=5,
        covered_branch_count=1,
        missing_branch_count=1,
        total_branch_count=2,
        test_exit_code=0,
        duration_seconds=0.1,
    )

    return FunctionCoverageResult(
        source_file=source_file,
        test_file=test_file,
        function_name=function_name,
        start_line=1,
        end_line=5,
        line_coverage_percent=line_coverage_percent,
        branch_coverage_percent=50.0,
        covered_lines=(1, 2),
        missing_lines=(3, 4, 5),
        covered_branch_count=1,
        missing_branch_count=1,
        test_exit_code=0,
        duration_seconds=0.1,
        file_coverage=file_coverage,
    )


def create_reachability_result(
    coverage_result: FunctionCoverageResult | None = None,
) -> FunctionCoverageReachabilityResult:
    """Dört satır durumunu içeren bounded analiz sonucu üretir."""
    normalized_coverage = (
        coverage_result
        if coverage_result is not None
        else create_function_coverage()
    )

    return FunctionCoverageReachabilityResult(
        coverage_result=normalized_coverage,
        line_evidence=(
            LineReachabilityEvidence(
                line_number=1,
                status=LineReachabilityStatus.COVERED,
            ),
            LineReachabilityEvidence(
                line_number=2,
                status=LineReachabilityStatus.COVERED,
            ),
            LineReachabilityEvidence(
                line_number=3,
                status=(
                    LineReachabilityStatus.FEASIBLE_UNCOVERED
                ),
                feasible_path_indices=(1,),
            ),
            LineReachabilityEvidence(
                line_number=4,
                status=LineReachabilityStatus.INFEASIBLE_ONLY,
                infeasible_path_indices=(2,),
            ),
            LineReachabilityEvidence(
                line_number=5,
                status=LineReachabilityStatus.UNRESOLVED,
                unknown_path_indices=(3,),
            ),
        ),
        analyzed_path_count=3,
        max_visits_per_node=3,
        path_metadata_complete=True,
    )


def format_report(
    *,
    coverage_result: FunctionCoverageResult | None = None,
    reachability_result: (
        FunctionCoverageReachabilityResult | None
    ) = None,
    function_name: str | None = "classify",
) -> str:
    """Geçerli oturum bilgileriyle rapor üretir."""
    result, statistics = create_training_results()

    return TrainingReportFormatter().format_session(
        result=result,
        statistics=statistics,
        function_name=function_name,
        coverage_result=coverage_result,
        reachability_result=reachability_result,
    )


def test_format_session_preserves_legacy_output() -> None:
    report = format_report(
        function_name=None,
    )

    assert "RL EĞİTİM OTURUMU" in report
    assert "Episode 1" in report
    assert "GENEL ÖZET" in report
    assert "SON COVERAGE ÖZETİ" not in report
    assert "SENARYO HAVUZU VE BOUNDED PATH ÖZETİ" not in report


def test_format_session_reports_final_coverage() -> None:
    report = format_report(
        coverage_result=create_function_coverage(),
    )

    assert "SON COVERAGE ÖZETİ" in report
    assert "Hedef fonksiyon              : classify" in report
    assert "Fonksiyon satır coverage      : %40.00" in report
    assert "Fonksiyon branch coverage     : %50.00" in report


def test_format_session_reports_bounded_reachability() -> None:
    reachability_result = create_reachability_result()

    report = format_report(
        coverage_result=create_function_coverage(
            line_coverage_percent=80.0,
        ),
        reachability_result=reachability_result,
    )

    assert "SENARYO HAVUZU VE BOUNDED PATH ÖZETİ" in report
    assert "Senaryo havuzu satır coverage : %40.00" in report
    assert "Analiz edilen bounded path    : 3" in report
    assert "Düğüm ziyaret sınırı          : 3" in report
    assert "Path metadata durumu          : Tam" in report
    assert "Kapsanan satırlar" in report
    assert "2 (1, 2)" in report
    assert "FEASIBLE fakat kapsanmayan" in report
    assert "1 (3)" in report
    assert "Yalnızca bounded INFEASIBLE" in report
    assert "1 (4)" in report
    assert "Çözümlenemeyen satırlar" in report
    assert "1 (5)" in report
    assert "Bounded sınıflandırma tamam mı : Hayır" in report


def test_format_session_distinguishes_final_and_pool_coverage() -> None:
    report = format_report(
        coverage_result=create_function_coverage(
            line_coverage_percent=80.0,
        ),
        reachability_result=create_reachability_result(),
    )

    assert "Fonksiyon satır coverage      : %80.00" in report
    assert "Senaryo havuzu satır coverage : %40.00" in report


def test_format_session_explains_infeasible_only_scope() -> None:
    report = format_report(
        reachability_result=create_reachability_result(),
    )

    assert (
        "INFEASIBLE_ONLY yalnızca analiz edilen bounded path "
        "kümesi için geçerlidir."
        in report
    )


def test_format_session_accepts_reachability_without_final_coverage(
) -> None:
    report = format_report(
        reachability_result=create_reachability_result(),
    )

    assert "SON COVERAGE ÖZETİ" not in report
    assert "SENARYO HAVUZU VE BOUNDED PATH ÖZETİ" in report


def test_format_session_rejects_invalid_reachability_type() -> None:
    result, statistics = create_training_results()

    with pytest.raises(
        TypeError,
        match=(
            "reachability_result bir "
            "FunctionCoverageReachabilityResult"
        ),
    ):
        TrainingReportFormatter().format_session(
            result=result,
            statistics=statistics,
            reachability_result="invalid",  # type: ignore[arg-type]
        )


def test_format_session_rejects_mismatching_function_name() -> None:
    with pytest.raises(
        ValueError,
        match="farklı bir hedef fonksiyona aittir",
    ):
        format_report(
            function_name="other_function",
            reachability_result=create_reachability_result(),
        )


def test_format_session_rejects_file_coverage_with_reachability() -> None:
    reachability_result = create_reachability_result()

    with pytest.raises(
        ValueError,
        match="yalnızca fonksiyon bazlı coverage",
    ):
        format_report(
            coverage_result=(
                reachability_result
                .coverage_result
                .file_coverage
            ),  # type: ignore[arg-type]
            reachability_result=reachability_result,
        )


def test_format_session_rejects_different_source_results() -> None:
    reachability_result = create_reachability_result()

    with pytest.raises(
        ValueError,
        match="aynı kaynak fonksiyona ait olmalıdır",
    ):
        format_report(
            coverage_result=create_function_coverage(
                source_file=Path("datasets/other.py"),
            ),
            reachability_result=reachability_result,
        )


def test_format_session_rejects_different_function_results() -> None:
    reachability_result = create_reachability_result()

    with pytest.raises(
        ValueError,
        match="aynı kaynak fonksiyona ait olmalıdır",
    ):
        format_report(
            function_name=None,
            coverage_result=create_function_coverage(
                function_name="other_function",
            ),
            reachability_result=reachability_result,
        )
