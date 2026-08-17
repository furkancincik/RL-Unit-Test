from __future__ import annotations

from pathlib import Path

import pytest

from cfg.path_analyzer import ExecutionPath
from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathFeasibilityResult,
)
from models.coverage_reachability_result import (
    FunctionCoverageReachabilityResult,
    LineReachabilityStatus,
)
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from services.coverage_reachability_service import (
    CoverageReachabilityService,
)


def create_coverage_result() -> FunctionCoverageResult:
    """Altı çalıştırılabilir satırlı fonksiyon coverage sonucu üretir."""
    source_file = Path("datasets/sample.py")
    test_file = Path("output/test_sample.py")

    file_coverage = CoverageResult(
        source_file=source_file,
        test_file=test_file,
        line_coverage_percent=33.33,
        branch_coverage_percent=50.0,
        covered_line_count=2,
        missing_line_count=4,
        total_line_count=6,
        covered_branch_count=1,
        missing_branch_count=1,
        total_branch_count=2,
        test_exit_code=0,
        duration_seconds=0.1,
    )

    return FunctionCoverageResult(
        source_file=source_file,
        test_file=test_file,
        function_name="sample",
        start_line=1,
        end_line=6,
        line_coverage_percent=33.33,
        branch_coverage_percent=50.0,
        covered_lines=(1, 2),
        missing_lines=(3, 4, 5, 6),
        covered_branch_count=1,
        missing_branch_count=1,
        test_exit_code=0,
        duration_seconds=0.1,
        file_coverage=file_coverage,
        covered_branches=((2, 3),),
        missing_branches=((2, 6),),
    )


def create_path(
    *line_numbers: object,
) -> ExecutionPath:
    """Verilen satır metadata'sını içeren yürütme yolu oluşturur."""
    node_count = len(line_numbers)

    return ExecutionPath(
        node_ids=list(range(1, node_count + 1)),
        edge_labels=[None] * (node_count - 1),
        node_labels=[
            f"line {line_number}"
            for line_number in line_numbers
        ],
        node_types=["Assign"] * node_count,
        line_numbers=list(line_numbers),  # type: ignore[arg-type]
    )


def create_path_without_metadata() -> ExecutionPath:
    """Eski API biçiminde metadata içermeyen yürütme yolu oluşturur."""
    return ExecutionPath(
        node_ids=[1],
        edge_labels=[],
    )


def create_feasibility_result(
    status: FeasibilityStatus,
) -> PathFeasibilityResult:
    """Belirtilen durumda boş constraint sonucu oluşturur."""
    return PathFeasibilityResult(
        status=status,
        constraints=(),
        conflicts=(),
    )


def analyze_default() -> FunctionCoverageReachabilityResult:
    """Dört sınıflandırma durumunu üreten örnek analiz çalıştırır."""
    return CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(1, 2, 3),
            create_path(1, 2, 4),
            create_path(1, 2, 5),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.UNKNOWN
            ),
        ),
        max_visits_per_node=3,
    )


def test_analyze_returns_reachability_result() -> None:
    result = analyze_default()

    assert isinstance(
        result,
        FunctionCoverageReachabilityResult,
    )


def test_analyze_classifies_all_line_groups() -> None:
    result = analyze_default()

    assert result.covered_lines == (1, 2)
    assert result.feasible_uncovered_lines == (3,)
    assert result.infeasible_only_lines == (4,)
    assert result.unresolved_lines == (5, 6)


def test_analyze_preserves_path_evidence() -> None:
    result = analyze_default()

    assert (
        result.evidence_for_line(3).feasible_path_indices
        == (1,)
    )
    assert (
        result.evidence_for_line(4).infeasible_path_indices
        == (2,)
    )
    assert (
        result.evidence_for_line(5).unknown_path_indices
        == (3,)
    )
    assert result.evidence_for_line(6).path_indices == ()


def test_covered_status_overrides_static_path_status() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(1, 2),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
        ),
        max_visits_per_node=2,
    )

    assert result.evidence_for_line(1).status == (
        LineReachabilityStatus.COVERED
    )
    assert result.evidence_for_line(2).status == (
        LineReachabilityStatus.COVERED
    )


def test_feasible_status_has_priority_for_missing_line() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(3),
            create_path(3),
            create_path(3),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.UNKNOWN
            ),
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
        ),
        max_visits_per_node=2,
    )

    evidence = result.evidence_for_line(3)

    assert evidence.status == (
        LineReachabilityStatus.FEASIBLE_UNCOVERED
    )
    assert evidence.feasible_path_indices == (3,)
    assert evidence.infeasible_path_indices == (1,)
    assert evidence.unknown_path_indices == (2,)


def test_unknown_status_prevents_infeasible_only_decision() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(4),
            create_path(4),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.UNKNOWN
            ),
        ),
        max_visits_per_node=2,
    )

    assert result.evidence_for_line(4).status == (
        LineReachabilityStatus.UNRESOLVED
    )


def test_only_infeasible_paths_produce_infeasible_only() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(4),
            create_path(4),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
        ),
        max_visits_per_node=2,
    )

    evidence = result.evidence_for_line(4)

    assert evidence.status == (
        LineReachabilityStatus.INFEASIBLE_ONLY
    )
    assert evidence.infeasible_path_indices == (1, 2)


def test_unmapped_missing_line_is_unresolved() -> None:
    result = analyze_default()

    evidence = result.evidence_for_line(6)

    assert evidence.status == (
        LineReachabilityStatus.UNRESOLVED
    )
    assert evidence.path_indices == ()


def test_missing_metadata_prevents_infeasible_only() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(4),
            create_path_without_metadata(),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
        ),
        max_visits_per_node=2,
    )

    assert result.path_metadata_complete is False
    assert result.infeasible_only_lines == ()
    assert result.evidence_for_line(4).status == (
        LineReachabilityStatus.UNRESOLVED
    )


def test_missing_metadata_does_not_erase_feasible_evidence() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(3),
            create_path_without_metadata(),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.UNKNOWN
            ),
        ),
        max_visits_per_node=2,
    )

    assert result.path_metadata_complete is False
    assert result.evidence_for_line(3).status == (
        LineReachabilityStatus.FEASIBLE_UNCOVERED
    )


@pytest.mark.parametrize(
    "invalid_line_number",
    (
        True,
        "4",
        0,
        -1,
    ),
)
def test_invalid_path_line_metadata_is_treated_as_incomplete(
    invalid_line_number: object,
) -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(invalid_line_number),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
        ),
        max_visits_per_node=2,
    )

    assert result.path_metadata_complete is False
    assert result.infeasible_only_lines == ()


def test_repeated_loop_line_is_counted_once_per_path() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(3, 3, 3),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
        ),
        max_visits_per_node=3,
    )

    assert (
        result.evidence_for_line(3).feasible_path_indices
        == (1,)
    )


def test_path_indices_preserve_input_order() -> None:
    result = CoverageReachabilityService().analyze(
        coverage_result=create_coverage_result(),
        paths=(
            create_path(3),
            create_path(3),
            create_path(3),
        ),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.INFEASIBLE
            ),
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
        ),
        max_visits_per_node=3,
    )

    evidence = result.evidence_for_line(3)

    assert evidence.feasible_path_indices == (1, 3)
    assert evidence.infeasible_path_indices == (2,)


def test_analyze_preserves_scope_and_coverage_result() -> None:
    coverage_result = create_coverage_result()

    result = CoverageReachabilityService().analyze(
        coverage_result=coverage_result,
        paths=(create_path(1, 2, 3),),
        feasibility_results=(
            create_feasibility_result(
                FeasibilityStatus.FEASIBLE
            ),
        ),
        max_visits_per_node=4,
    )

    assert result.coverage_result is coverage_result
    assert result.analyzed_path_count == 1
    assert result.max_visits_per_node == 4


def test_analyze_rejects_invalid_coverage_result() -> None:
    with pytest.raises(
        TypeError,
        match="coverage_result bir FunctionCoverageResult",
    ):
        CoverageReachabilityService().analyze(
            coverage_result="invalid",  # type: ignore[arg-type]
            paths=(create_path(1),),
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=2,
        )


def test_analyze_rejects_non_tuple_paths() -> None:
    with pytest.raises(
        TypeError,
        match="paths bir ExecutionPath tuple'ı olmalıdır",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=[create_path(1)],  # type: ignore[arg-type]
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=2,
        )


def test_analyze_rejects_empty_paths() -> None:
    with pytest.raises(
        ValueError,
        match="paths boş olamaz",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(),
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=2,
        )


def test_analyze_rejects_invalid_path_item() -> None:
    with pytest.raises(
        TypeError,
        match="paths yalnızca ExecutionPath",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=("invalid",),  # type: ignore[arg-type]
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=2,
        )


def test_analyze_rejects_non_tuple_feasibility_results() -> None:
    with pytest.raises(
        TypeError,
        match="feasibility_results bir PathFeasibilityResult",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(create_path(1),),
            feasibility_results=[],  # type: ignore[arg-type]
            max_visits_per_node=2,
        )


def test_analyze_rejects_empty_feasibility_results() -> None:
    with pytest.raises(
        ValueError,
        match="feasibility_results boş olamaz",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(create_path(1),),
            feasibility_results=(),
            max_visits_per_node=2,
        )


def test_analyze_rejects_invalid_feasibility_item() -> None:
    with pytest.raises(
        TypeError,
        match="yalnızca PathFeasibilityResult",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(create_path(1),),
            feasibility_results=("invalid",),  # type: ignore[arg-type]
            max_visits_per_node=2,
        )


def test_analyze_rejects_invalid_feasibility_status() -> None:
    invalid_result = PathFeasibilityResult(
        status="INVALID",  # type: ignore[arg-type]
        constraints=(),
        conflicts=(),
    )

    with pytest.raises(
        TypeError,
        match="status alanları FeasibilityStatus",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(create_path(1),),
            feasibility_results=(invalid_result,),
            max_visits_per_node=2,
        )


def test_analyze_rejects_path_result_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="aynı uzunlukta olmalıdır",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(
                create_path(1),
                create_path(2),
            ),
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=2,
        )


@pytest.mark.parametrize(
    "max_visits_per_node",
    (
        True,
        2.0,
        "2",
    ),
)
def test_analyze_rejects_invalid_max_visits_type(
    max_visits_per_node: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="max_visits_per_node bir integer olmalıdır",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(create_path(1),),
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=max_visits_per_node,  # type: ignore[arg-type]
        )


def test_analyze_rejects_non_positive_max_visits() -> None:
    with pytest.raises(
        ValueError,
        match="max_visits_per_node en az 1 olmalıdır",
    ):
        CoverageReachabilityService().analyze(
            coverage_result=create_coverage_result(),
            paths=(create_path(1),),
            feasibility_results=(
                create_feasibility_result(
                    FeasibilityStatus.FEASIBLE
                ),
            ),
            max_visits_per_node=0,
        )
