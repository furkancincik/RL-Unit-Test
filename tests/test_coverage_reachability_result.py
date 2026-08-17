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


def create_function_coverage() -> FunctionCoverageResult:
    """Reachability model testleri için fonksiyon coverage sonucu üretir."""
    source_file = Path("datasets/sample.py")
    test_file = Path("output/test_sample.py")

    file_coverage = CoverageResult(
        source_file=source_file,
        test_file=test_file,
        line_coverage_percent=40.0,
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
        function_name="sample",
        start_line=1,
        end_line=5,
        line_coverage_percent=40.0,
        branch_coverage_percent=50.0,
        covered_lines=(1, 2),
        missing_lines=(3, 4, 5),
        covered_branch_count=1,
        missing_branch_count=1,
        test_exit_code=0,
        duration_seconds=0.1,
        file_coverage=file_coverage,
        covered_branches=((2, 3),),
        missing_branches=((2, 5),),
    )


def create_line_evidence() -> tuple[LineReachabilityEvidence, ...]:
    """Dört sınıflandırma durumunu içeren satır kanıtları oluşturur."""
    return (
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(1, 2),
        ),
        LineReachabilityEvidence(
            line_number=2,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(1,),
        ),
        LineReachabilityEvidence(
            line_number=3,
            status=LineReachabilityStatus.FEASIBLE_UNCOVERED,
            feasible_path_indices=(2,),
            infeasible_path_indices=(3,),
        ),
        LineReachabilityEvidence(
            line_number=4,
            status=LineReachabilityStatus.INFEASIBLE_ONLY,
            infeasible_path_indices=(3,),
        ),
        LineReachabilityEvidence(
            line_number=5,
            status=LineReachabilityStatus.UNRESOLVED,
            infeasible_path_indices=(3,),
            unknown_path_indices=(4,),
        ),
    )


def create_result(
    **overrides: object,
) -> FunctionCoverageReachabilityResult:
    """Geçerli reachability sonucu oluşturur."""
    values: dict[str, object] = {
        "coverage_result": create_function_coverage(),
        "line_evidence": create_line_evidence(),
        "analyzed_path_count": 4,
        "max_visits_per_node": 3,
        "path_metadata_complete": True,
    }
    values.update(overrides)

    return FunctionCoverageReachabilityResult(
        **values,  # type: ignore[arg-type]
    )


def test_line_reachability_status_values_are_stable() -> None:
    assert LineReachabilityStatus.COVERED.value == "COVERED"
    assert (
        LineReachabilityStatus.FEASIBLE_UNCOVERED.value
        == "FEASIBLE_UNCOVERED"
    )
    assert (
        LineReachabilityStatus.INFEASIBLE_ONLY.value
        == "INFEASIBLE_ONLY"
    )
    assert LineReachabilityStatus.UNRESOLVED.value == "UNRESOLVED"


def test_line_evidence_combines_path_indices() -> None:
    evidence = LineReachabilityEvidence(
        line_number=3,
        status=LineReachabilityStatus.COVERED,
        feasible_path_indices=(1, 4),
        infeasible_path_indices=(2,),
        unknown_path_indices=(3,),
    )

    assert evidence.path_indices == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "line_number",
    (
        True,
        "3",
        3.0,
    ),
)
def test_line_evidence_rejects_invalid_line_type(
    line_number: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="line_number bir integer olmalıdır",
    ):
        LineReachabilityEvidence(
            line_number=line_number,  # type: ignore[arg-type]
            status=LineReachabilityStatus.COVERED,
        )


def test_line_evidence_rejects_non_positive_line() -> None:
    with pytest.raises(
        ValueError,
        match="line_number en az 1 olmalıdır",
    ):
        LineReachabilityEvidence(
            line_number=0,
            status=LineReachabilityStatus.COVERED,
        )


def test_line_evidence_rejects_invalid_status() -> None:
    with pytest.raises(
        TypeError,
        match="status bir LineReachabilityStatus",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status="COVERED",  # type: ignore[arg-type]
        )


def test_line_evidence_rejects_non_tuple_path_indices() -> None:
    with pytest.raises(
        TypeError,
        match="feasible_path_indices bir tuple olmalıdır",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=[1],  # type: ignore[arg-type]
        )


def test_line_evidence_rejects_invalid_path_index_type() -> None:
    with pytest.raises(
        TypeError,
        match="yalnızca integer değerler içermelidir",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(True,),
        )


def test_line_evidence_rejects_non_positive_path_index() -> None:
    with pytest.raises(
        ValueError,
        match="yalnızca pozitif değerler içermelidir",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(0,),
        )


def test_line_evidence_rejects_duplicate_path_indices() -> None:
    with pytest.raises(
        ValueError,
        match="tekrar eden path indeksi içeremez",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(1, 1),
        )


def test_line_evidence_rejects_unsorted_path_indices() -> None:
    with pytest.raises(
        ValueError,
        match="artan sırada olmalıdır",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(2, 1),
        )


def test_line_evidence_rejects_overlapping_status_indices() -> None:
    with pytest.raises(
        ValueError,
        match="birden fazla feasibility durumunda bulunamaz",
    ):
        LineReachabilityEvidence(
            line_number=1,
            status=LineReachabilityStatus.COVERED,
            feasible_path_indices=(1,),
            unknown_path_indices=(1,),
        )


def test_feasible_uncovered_requires_feasible_path() -> None:
    with pytest.raises(
        ValueError,
        match="en az bir FEASIBLE path kanıtı",
    ):
        LineReachabilityEvidence(
            line_number=3,
            status=LineReachabilityStatus.FEASIBLE_UNCOVERED,
        )


def test_infeasible_only_requires_infeasible_path() -> None:
    with pytest.raises(
        ValueError,
        match="en az bir INFEASIBLE path kanıtı",
    ):
        LineReachabilityEvidence(
            line_number=4,
            status=LineReachabilityStatus.INFEASIBLE_ONLY,
        )


def test_infeasible_only_rejects_unknown_path() -> None:
    with pytest.raises(
        ValueError,
        match="FEASIBLE veya UNKNOWN path kanıtı içeremez",
    ):
        LineReachabilityEvidence(
            line_number=4,
            status=LineReachabilityStatus.INFEASIBLE_ONLY,
            infeasible_path_indices=(1,),
            unknown_path_indices=(2,),
        )


def test_unresolved_rejects_feasible_path() -> None:
    with pytest.raises(
        ValueError,
        match="UNRESOLVED satır FEASIBLE path kanıtı içeremez",
    ):
        LineReachabilityEvidence(
            line_number=5,
            status=LineReachabilityStatus.UNRESOLVED,
            feasible_path_indices=(1,),
        )


def test_result_exposes_classified_line_groups() -> None:
    result = create_result()

    assert result.covered_lines == (1, 2)
    assert result.feasible_uncovered_lines == (3,)
    assert result.infeasible_only_lines == (4,)
    assert result.unresolved_lines == (5,)
    assert result.covered_line_count == 2
    assert result.feasible_uncovered_line_count == 1
    assert result.infeasible_only_line_count == 1
    assert result.unresolved_line_count == 1
    assert result.has_unresolved_lines is True
    assert result.classification_complete is False
    assert result.observed_line_coverage_percent == 40.0


def test_result_reports_complete_classification() -> None:
    evidence = list(create_line_evidence())
    evidence[-1] = LineReachabilityEvidence(
        line_number=5,
        status=LineReachabilityStatus.INFEASIBLE_ONLY,
        infeasible_path_indices=(3, 4),
    )

    result = create_result(
        line_evidence=tuple(evidence),
    )

    assert result.has_unresolved_lines is False
    assert result.classification_complete is True


def test_result_returns_evidence_for_line() -> None:
    result = create_result()

    evidence = result.evidence_for_line(3)

    assert evidence.status == (
        LineReachabilityStatus.FEASIBLE_UNCOVERED
    )
    assert evidence.feasible_path_indices == (2,)


def test_result_rejects_unknown_evidence_line() -> None:
    with pytest.raises(
        KeyError,
        match="Coverage sonucunda satır bulunamadı: 99",
    ):
        create_result().evidence_for_line(99)


def test_result_rejects_invalid_evidence_lookup_type() -> None:
    with pytest.raises(
        TypeError,
        match="line_number bir integer olmalıdır",
    ):
        create_result().evidence_for_line(True)


def test_result_rejects_invalid_coverage_result() -> None:
    with pytest.raises(
        TypeError,
        match="coverage_result bir FunctionCoverageResult",
    ):
        create_result(
            coverage_result="invalid",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "analyzed_path_count",
        "max_visits_per_node",
    ),
)
def test_result_rejects_non_positive_analysis_integer(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} en az 1 olmalıdır",
    ):
        create_result(
            **{
                field_name: 0,
            }
        )


def test_result_rejects_invalid_metadata_flag() -> None:
    with pytest.raises(
        TypeError,
        match="path_metadata_complete bir bool olmalıdır",
    ):
        create_result(
            path_metadata_complete=1,
        )


def test_result_rejects_non_tuple_evidence() -> None:
    with pytest.raises(
        TypeError,
        match="line_evidence bir tuple olmalıdır",
    ):
        create_result(
            line_evidence=list(create_line_evidence()),
        )


def test_result_rejects_invalid_evidence_item() -> None:
    with pytest.raises(
        TypeError,
        match="yalnızca LineReachabilityEvidence",
    ):
        create_result(
            line_evidence=(
                *create_line_evidence()[:-1],
                "invalid",
            ),
        )


def test_result_rejects_duplicate_evidence_line() -> None:
    evidence = create_line_evidence()

    with pytest.raises(
        ValueError,
        match="tekrar eden satır içeremez",
    ):
        create_result(
            line_evidence=(
                *evidence,
                evidence[-1],
            ),
        )


def test_result_rejects_unsorted_evidence_lines() -> None:
    with pytest.raises(
        ValueError,
        match="satır numarasına göre artan sırada olmalıdır",
    ):
        create_result(
            line_evidence=tuple(
                reversed(create_line_evidence())
            ),
        )


def test_result_requires_exact_coverage_line_partition() -> None:
    with pytest.raises(
        ValueError,
        match="çalıştırılabilir satırları tam olarak içermelidir",
    ):
        create_result(
            line_evidence=create_line_evidence()[:-1],
        )


def test_result_requires_covered_status_for_executed_line() -> None:
    evidence = list(create_line_evidence())
    evidence[0] = LineReachabilityEvidence(
        line_number=1,
        status=LineReachabilityStatus.FEASIBLE_UNCOVERED,
        feasible_path_indices=(1,),
    )

    with pytest.raises(
        ValueError,
        match="çalıştırılmış satırlar COVERED",
    ):
        create_result(
            line_evidence=tuple(evidence),
        )


def test_result_rejects_covered_status_for_missing_line() -> None:
    evidence = list(create_line_evidence())
    evidence[2] = LineReachabilityEvidence(
        line_number=3,
        status=LineReachabilityStatus.COVERED,
    )

    with pytest.raises(
        ValueError,
        match="eksik bildirilen satırlar COVERED",
    ):
        create_result(
            line_evidence=tuple(evidence),
        )


def test_result_rejects_path_index_above_analyzed_count() -> None:
    evidence = list(create_line_evidence())
    evidence[2] = LineReachabilityEvidence(
        line_number=3,
        status=LineReachabilityStatus.FEASIBLE_UNCOVERED,
        feasible_path_indices=(5,),
    )

    with pytest.raises(
        ValueError,
        match="path sayısını aşamaz",
    ):
        create_result(
            line_evidence=tuple(evidence),
        )


def test_result_rejects_infeasible_only_when_metadata_is_incomplete() -> None:
    with pytest.raises(
        ValueError,
        match="metadata eksikken INFEASIBLE_ONLY",
    ):
        create_result(
            path_metadata_complete=False,
        )


def test_result_accepts_unresolved_when_metadata_is_incomplete() -> None:
    evidence = list(create_line_evidence())
    evidence[3] = LineReachabilityEvidence(
        line_number=4,
        status=LineReachabilityStatus.UNRESOLVED,
        infeasible_path_indices=(3,),
    )

    result = create_result(
        line_evidence=tuple(evidence),
        path_metadata_complete=False,
    )

    assert result.path_metadata_complete is False
    assert result.classification_complete is False
    assert result.unresolved_lines == (4, 5)
