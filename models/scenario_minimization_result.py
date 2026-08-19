from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CoverageMeasurementStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    INTERACTION_DEPENDENT = "INTERACTION_DEPENDENT"


class ScenarioMinimizationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    NO_REDUCTION = "NO_REDUCTION"
    EMPTY_POOL = "EMPTY_POOL"
    FULL_POOL_FAILED = "FULL_POOL_FAILED"
    TARGET_NOT_COVERED = "TARGET_NOT_COVERED"
    NON_ADDITIVE_COVERAGE = "NON_ADDITIVE_COVERAGE"
    TIMED_OUT = "TIMED_OUT"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True, slots=True)
class ScenarioCoverageSignature:
    """Tek scenario'nun güvenli function-level coverage katkısıdır."""

    scenario_id: str
    original_scenario_index: int
    function_name: str
    covered_line_identities: tuple[int, ...]
    covered_branch_identities: tuple[tuple[int, int], ...]
    line_contribution_count: int
    branch_contribution_count: int
    execution_success: bool
    test_exit_code: int | None
    duration_seconds: float | None
    measurement_status: CoverageMeasurementStatus
    failure_category: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not isinstance(self.scenario_id, str):
            raise ValueError("scenario_id boş olamaz.")
        if (
            isinstance(self.original_scenario_index, bool)
            or not isinstance(self.original_scenario_index, int)
            or self.original_scenario_index < 1
        ):
            raise ValueError("original_scenario_index pozitif tam sayı olmalıdır.")
        if not isinstance(self.function_name, str) or not self.function_name:
            raise ValueError("function_name boş olamaz.")
        self._validate_lines(self.covered_line_identities)
        self._validate_branches(self.covered_branch_identities)
        if self.line_contribution_count != len(self.covered_line_identities):
            raise ValueError("line_contribution_count identity sayısıyla uyuşmuyor.")
        if self.branch_contribution_count != len(self.covered_branch_identities):
            raise ValueError("branch_contribution_count identity sayısıyla uyuşmuyor.")
        if not isinstance(self.execution_success, bool):
            raise TypeError("execution_success bool olmalıdır.")
        if self.test_exit_code is not None and (
            isinstance(self.test_exit_code, bool)
            or not isinstance(self.test_exit_code, int)
        ):
            raise TypeError("test_exit_code tam sayı veya None olmalıdır.")
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0.0
        ):
            raise ValueError("duration_seconds negatif olmayan sonlu sayı olmalıdır.")
        if not isinstance(self.measurement_status, CoverageMeasurementStatus):
            raise TypeError("measurement_status geçersiz.")

    @staticmethod
    def _validate_lines(lines: tuple[int, ...]) -> None:
        if not isinstance(lines, tuple):
            raise TypeError("covered_line_identities tuple olmalıdır.")
        if any(isinstance(line, bool) or not isinstance(line, int) or line < 1 for line in lines):
            raise ValueError("covered_line_identities pozitif tam sayılar içermelidir.")
        if tuple(sorted(set(lines))) != lines:
            raise ValueError("covered_line_identities sıralı ve benzersiz olmalıdır.")

    @staticmethod
    def _validate_branches(branches: tuple[tuple[int, int], ...]) -> None:
        if not isinstance(branches, tuple):
            raise TypeError("covered_branch_identities tuple olmalıdır.")
        for branch in branches:
            if not isinstance(branch, tuple) or len(branch) != 2:
                raise TypeError("Branch identity iki elemanlı tuple olmalıdır.")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in branch):
                raise TypeError("Branch identity tam sayılardan oluşmalıdır.")
            if branch[0] < 1:
                raise ValueError("Branch kaynak satırı pozitif olmalıdır.")
        if tuple(sorted(set(branches))) != branches:
            raise ValueError("covered_branch_identities sıralı ve benzersiz olmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "original_scenario_index": self.original_scenario_index,
            "function_name": self.function_name,
            "covered_line_identities": list(self.covered_line_identities),
            "covered_branch_identities": [list(branch) for branch in self.covered_branch_identities],
            "line_contribution_count": self.line_contribution_count,
            "branch_contribution_count": self.branch_contribution_count,
            "execution_success": self.execution_success,
            "test_exit_code": self.test_exit_code,
            "duration_seconds": self.duration_seconds,
            "measurement_status": self.measurement_status.value,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True, slots=True)
class ScenarioMinimizationResult:
    """Exact coverage identity tabanlı greedy baseline sonucudur."""

    source_file: Path
    function_name: str
    status: ScenarioMinimizationStatus
    full_pool_scenario_count: int
    contributions: tuple[ScenarioCoverageSignature, ...]
    target_line_identities: tuple[int, ...]
    target_branch_identities: tuple[tuple[int, int], ...]
    target_line_percentage: float | None
    target_branch_percentage: float | None
    greedy_initially_selected_count: int
    redundancy_elimination_removed_count: int
    final_selected_scenario_ids: tuple[str, ...]
    removed_scenario_ids: tuple[str, ...]
    coverage_preserved: bool
    final_verified_line_identities: tuple[int, ...]
    final_verified_branch_identities: tuple[tuple[int, int], ...]
    final_verified_line_percentage: float | None
    final_verified_branch_percentage: float | None
    duration_seconds: float
    failure_category: str | None = None
    failure_message: str | None = None
    artifact_paths: tuple[Path, ...] = ()
    algorithm: str = "GREEDY_SET_COVER_WITH_BACKWARD_ELIMINATION"
    globally_minimal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, Path):
            raise TypeError("source_file Path olmalıdır.")
        if not isinstance(self.status, ScenarioMinimizationStatus):
            raise TypeError("status geçersiz.")
        for name in (
            "full_pool_scenario_count",
            "greedy_initially_selected_count",
            "redundancy_elimination_removed_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")
        if self.globally_minimal is not False:
            raise ValueError("Greedy baseline globally_minimal olamaz.")
        if self.algorithm != "GREEDY_SET_COVER_WITH_BACKWARD_ELIMINATION":
            raise ValueError("algorithm desteklenmiyor.")
        if self.final_selected_count > self.full_pool_scenario_count:
            raise ValueError("Final scenario sayısı full pool sayısını aşamaz.")
        if self.final_selected_count + self.reduction_count > self.full_pool_scenario_count:
            raise ValueError("Selected ve removed scenario sayıları full pool'u aşamaz.")
        if len(self.contributions) > self.full_pool_scenario_count:
            raise ValueError("Contribution sayısı full pool sayısını aşamaz.")
        contribution_ids = tuple(item.scenario_id for item in self.contributions)
        if len(set(contribution_ids)) != len(contribution_ids):
            raise ValueError("Contribution scenario ID değerleri benzersiz olmalıdır.")
        if len(set(self.final_selected_scenario_ids)) != self.final_selected_count:
            raise ValueError("Final selected scenario ID değerleri benzersiz olmalıdır.")
        if len(set(self.removed_scenario_ids)) != self.reduction_count:
            raise ValueError("Removed scenario ID değerleri benzersiz olmalıdır.")
        if set(self.final_selected_scenario_ids) & set(self.removed_scenario_ids):
            raise ValueError("Selected ve removed scenario kümeleri ayrık olmalıdır.")
        for percentage_name in (
            "target_line_percentage",
            "target_branch_percentage",
            "final_verified_line_percentage",
            "final_verified_branch_percentage",
        ):
            percentage = getattr(self, percentage_name)
            if percentage is not None and (
                isinstance(percentage, bool)
                or not isinstance(percentage, (int, float))
                or not math.isfinite(float(percentage))
                or not 0.0 <= float(percentage) <= 100.0
            ):
                raise ValueError(f"{percentage_name} 0 ile 100 arasında olmalıdır.")
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds geçersiz.")

    @property
    def measured_scenario_count(self) -> int:
        return len(self.contributions)

    @property
    def failed_measurement_count(self) -> int:
        return sum(not item.execution_success for item in self.contributions)

    @property
    def target_line_count(self) -> int:
        return len(self.target_line_identities)

    @property
    def target_branch_count(self) -> int:
        return len(self.target_branch_identities)

    @property
    def final_selected_count(self) -> int:
        return len(self.final_selected_scenario_ids)

    @property
    def reduction_count(self) -> int:
        return len(self.removed_scenario_ids)

    @property
    def reduction_percentage(self) -> float:
        if self.full_pool_scenario_count == 0:
            return 0.0
        return self.reduction_count / self.full_pool_scenario_count * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "source_file": str(self.source_file),
            "function_name": self.function_name,
            "algorithm": self.algorithm,
            "run_status": self.status.value,
            "full_pool_scenario_count": self.full_pool_scenario_count,
            "measured_scenario_count": self.measured_scenario_count,
            "failed_measurement_count": self.failed_measurement_count,
            "target_line_identities": list(self.target_line_identities),
            "target_line_count": self.target_line_count,
            "target_line_percentage": self.target_line_percentage,
            "target_branch_identities": [list(branch) for branch in self.target_branch_identities],
            "target_branch_count": self.target_branch_count,
            "target_branch_percentage": self.target_branch_percentage,
            "greedy_initially_selected_count": self.greedy_initially_selected_count,
            "redundancy_elimination_removed_count": self.redundancy_elimination_removed_count,
            "final_selected_scenario_ids": list(self.final_selected_scenario_ids),
            "final_selected_count": self.final_selected_count,
            "removed_scenario_ids": list(self.removed_scenario_ids),
            "removed_scenario_count": len(self.removed_scenario_ids),
            "reduction_count": self.reduction_count,
            "reduction_percentage": self.reduction_percentage,
            "coverage_preserved": self.coverage_preserved,
            "final_verified_line_identities": list(self.final_verified_line_identities),
            "final_verified_line_percentage": self.final_verified_line_percentage,
            "final_verified_branch_identities": [list(branch) for branch in self.final_verified_branch_identities],
            "final_verified_branch_percentage": self.final_verified_branch_percentage,
            "globally_minimal": self.globally_minimal,
            "duration_seconds": self.duration_seconds,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "contributions": [item.to_dict() for item in self.contributions],
        }
