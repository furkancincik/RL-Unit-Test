from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any

from models.project_coverage_result import (
    ProjectBranchIdentity,
    ProjectLineIdentity,
)


LineIdentity = tuple[str, int]
BranchIdentity = tuple[str, int, int]


class StrategyName(str, Enum):
    DQM_PREFIX = "DQM_PREFIX"
    EXACT_GREEDY = "EXACT_GREEDY"
    RL = "RL"


class StrategyStopReason(str, Enum):
    TARGET_REACHED = "TARGET_REACHED"
    CANDIDATES_EXHAUSTED = "CANDIDATES_EXHAUSTED"
    NO_FURTHER_CONTRIBUTION = "NO_FURTHER_CONTRIBUTION"


class StrategyWinnerOutcome(str, Enum):
    RL_WINS = "RL_WINS"
    BASELINE_WINS = "BASELINE_WINS"
    TIE = "TIE"
    MIXED = "MIXED"


def _validate_lines(values: tuple[LineIdentity, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} tuple olmalıdır.")
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} geçersiz line identity içeriyor.")
        ProjectLineIdentity(*value)
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} sıralı ve benzersiz olmalıdır.")


def _validate_branches(values: tuple[BranchIdentity, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} tuple olmalıdır.")
    for value in values:
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError(f"{name} geçersiz branch identity içeriyor.")
        ProjectBranchIdentity(*value)
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} sıralı ve benzersiz olmalıdır.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")


def _validate_finite_non_negative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} negatif olmayan sonlu sayı olmalıdır.")


@dataclass(frozen=True, slots=True)
class StrategyEvaluationCandidate:
    """Execution-derived exact contribution taşıyan güvenli aday özeti."""

    candidate_id: str
    dqm_rank: int
    covered_line_identities: tuple[LineIdentity, ...]
    covered_branch_identities: tuple[BranchIdentity, ...]
    execution_success: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id boş olmayan string olmalıdır.")
        if (
            isinstance(self.dqm_rank, bool)
            or not isinstance(self.dqm_rank, int)
            or self.dqm_rank < 1
        ):
            raise ValueError("dqm_rank pozitif tam sayı olmalıdır.")
        _validate_lines(self.covered_line_identities, "covered_line_identities")
        _validate_branches(
            self.covered_branch_identities,
            "covered_branch_identities",
        )
        if not isinstance(self.execution_success, bool):
            raise TypeError("execution_success bool olmalıdır.")


@dataclass(frozen=True, slots=True)
class StrategyEvaluationPool:
    """Üç strategy'nin değiştiremediği ortak exact coverage oracle'ıdır."""

    candidates: tuple[StrategyEvaluationCandidate, ...]
    executable_line_identities: tuple[LineIdentity, ...]
    executable_branch_identities: tuple[BranchIdentity, ...]
    target_line_identities: tuple[LineIdentity, ...]
    target_branch_identities: tuple[BranchIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates dolu tuple olmalıdır.")
        if any(
            not isinstance(candidate, StrategyEvaluationCandidate)
            for candidate in self.candidates
        ):
            raise TypeError("candidates yalnız StrategyEvaluationCandidate içermelidir.")
        identifiers = tuple(candidate.candidate_id for candidate in self.candidates)
        ranks = tuple(candidate.dqm_rank for candidate in self.candidates)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate_id değerleri benzersiz olmalıdır.")
        if len(set(ranks)) != len(ranks):
            raise ValueError("dqm_rank değerleri benzersiz olmalıdır.")
        if tuple(sorted(self.candidates, key=lambda item: item.dqm_rank)) != self.candidates:
            raise ValueError("candidates DQM sırasına göre verilmelidir.")
        if any(not candidate.execution_success for candidate in self.candidates):
            raise ValueError(
                "Ortak evaluation havuzuna yalnız execution_success adaylar girebilir."
            )

        _validate_lines(self.executable_line_identities, "executable_line_identities")
        _validate_branches(
            self.executable_branch_identities,
            "executable_branch_identities",
        )
        _validate_lines(self.target_line_identities, "target_line_identities")
        _validate_branches(self.target_branch_identities, "target_branch_identities")
        if not self.executable_line_identities and not self.executable_branch_identities:
            raise ValueError("Exact coverage denominator boş olamaz.")
        if not self.target_line_identities and not self.target_branch_identities:
            raise ValueError("Exact coverage target boş olamaz.")
        if not set(self.target_line_identities) <= set(self.executable_line_identities):
            raise ValueError("Target line identity denominator dışında olamaz.")
        if not set(self.target_branch_identities) <= set(
            self.executable_branch_identities
        ):
            raise ValueError("Target branch identity denominator dışında olamaz.")
        for candidate in self.candidates:
            if not set(candidate.covered_line_identities) <= set(
                self.target_line_identities
            ):
                raise ValueError("Candidate line contribution target dışında olamaz.")
            if not set(candidate.covered_branch_identities) <= set(
                self.target_branch_identities
            ):
                raise ValueError("Candidate branch contribution target dışında olamaz.")

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class StrategyCoverageStep:
    step_number: int
    candidate_id: str
    new_line_identities: tuple[LineIdentity, ...]
    new_branch_identities: tuple[BranchIdentity, ...]
    covered_line_identities: tuple[LineIdentity, ...]
    covered_branch_identities: tuple[BranchIdentity, ...]
    combined_coverage_percent: float
    reward: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_number, bool)
            or not isinstance(self.step_number, int)
            or self.step_number < 1
        ):
            raise ValueError("step_number pozitif tam sayı olmalıdır.")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id boş olamaz.")
        _validate_lines(self.new_line_identities, "new_line_identities")
        _validate_branches(self.new_branch_identities, "new_branch_identities")
        _validate_lines(self.covered_line_identities, "covered_line_identities")
        _validate_branches(self.covered_branch_identities, "covered_branch_identities")
        if not set(self.new_line_identities) <= set(self.covered_line_identities):
            raise ValueError("Yeni line identity cumulative küme dışında olamaz.")
        if not set(self.new_branch_identities) <= set(self.covered_branch_identities):
            raise ValueError("Yeni branch identity cumulative küme dışında olamaz.")
        if (
            isinstance(self.combined_coverage_percent, bool)
            or not isinstance(self.combined_coverage_percent, (int, float))
            or not math.isfinite(float(self.combined_coverage_percent))
            or not 0.0 <= float(self.combined_coverage_percent) <= 100.0
        ):
            raise ValueError("combined_coverage_percent geçersiz.")
        if self.reward is not None and (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(float(self.reward))
        ):
            raise ValueError("reward sonlu sayı veya None olmalıdır.")

    @property
    def exact_new_identity_count(self) -> int:
        return len(self.new_line_identities) + len(self.new_branch_identities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "candidate_id": self.candidate_id,
            "new_line_identities": [list(value) for value in self.new_line_identities],
            "new_branch_identities": [list(value) for value in self.new_branch_identities],
            "exact_new_identity_count": self.exact_new_identity_count,
            "covered_line_identities": [
                list(value) for value in self.covered_line_identities
            ],
            "covered_branch_identities": [
                list(value) for value in self.covered_branch_identities
            ],
            "combined_coverage_percent": self.combined_coverage_percent,
            "reward": self.reward,
        }


@dataclass(frozen=True, slots=True)
class StrategyEvaluationResult:
    strategy: StrategyName
    seed: int | None
    candidate_pool_count: int
    selected_candidate_ids: tuple[str, ...]
    executed_test_count: int
    effective_test_count: int
    zero_contribution_selection_count: int
    executable_line_identities: tuple[LineIdentity, ...]
    executable_branch_identities: tuple[BranchIdentity, ...]
    target_line_identities: tuple[LineIdentity, ...]
    target_branch_identities: tuple[BranchIdentity, ...]
    covered_line_identities: tuple[LineIdentity, ...]
    covered_branch_identities: tuple[BranchIdentity, ...]
    coverage_progression: tuple[StrategyCoverageStep, ...]
    tests_to_target: int | None
    stop_reason: StrategyStopReason
    duration_seconds: float
    episode_count: int | None = None
    q_state_count: int | None = None
    actions_evaluated: int | None = None
    exploration_selection_count: int | None = None
    exploitation_selection_count: int | None = None
    cumulative_reward: float | None = None
    final_reward: float | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, StrategyName):
            raise TypeError("strategy geçersiz.")
        if self.strategy is StrategyName.RL:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("RL seed tam sayı olmalıdır.")
        elif self.seed is not None:
            raise ValueError("Deterministic strategy seed taşıyamaz.")
        for name in (
            "candidate_pool_count",
            "executed_test_count",
            "effective_test_count",
            "zero_contribution_selection_count",
        ):
            _validate_non_negative_integer(getattr(self, name), name)
        if not isinstance(self.selected_candidate_ids, tuple) or any(
            not isinstance(value, str) or not value
            for value in self.selected_candidate_ids
        ):
            raise TypeError("selected_candidate_ids string tuple'ı olmalıdır.")
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise ValueError("Bir strategy aynı candidate'ı iki kez seçemez.")
        if self.executed_test_count != len(self.selected_candidate_ids):
            raise ValueError("executed_test_count selected count ile uyuşmuyor.")
        if self.executed_test_count > self.candidate_pool_count:
            raise ValueError("executed_test_count candidate pool'u aşamaz.")
        if (
            self.effective_test_count + self.zero_contribution_selection_count
            != self.executed_test_count
        ):
            raise ValueError("Effective ve zero contribution sayaçları tutarsız.")
        if len(self.coverage_progression) != self.executed_test_count:
            raise ValueError("coverage_progression executed count ile uyuşmuyor.")
        if tuple(step.step_number for step in self.coverage_progression) != tuple(
            range(1, self.executed_test_count + 1)
        ):
            raise ValueError("Coverage step numaraları ardışık olmalıdır.")
        if tuple(step.candidate_id for step in self.coverage_progression) != (
            self.selected_candidate_ids
        ):
            raise ValueError("Coverage progression candidate sırası tutarsız.")
        for name in (
            "executable_line_identities",
            "target_line_identities",
            "covered_line_identities",
        ):
            _validate_lines(getattr(self, name), name)
        for name in (
            "executable_branch_identities",
            "target_branch_identities",
            "covered_branch_identities",
        ):
            _validate_branches(getattr(self, name), name)
        if not set(self.target_line_identities) <= set(self.executable_line_identities):
            raise ValueError("Target line denominator dışında.")
        if not set(self.target_branch_identities) <= set(
            self.executable_branch_identities
        ):
            raise ValueError("Target branch denominator dışında.")
        if not set(self.covered_line_identities) <= set(self.target_line_identities):
            raise ValueError("Covered line target dışında.")
        if not set(self.covered_branch_identities) <= set(
            self.target_branch_identities
        ):
            raise ValueError("Covered branch target dışında.")
        if self.coverage_progression:
            last = self.coverage_progression[-1]
            if last.covered_line_identities != self.covered_line_identities:
                raise ValueError("Final line identities progression ile uyuşmuyor.")
            if last.covered_branch_identities != self.covered_branch_identities:
                raise ValueError("Final branch identities progression ile uyuşmuyor.")
        if self.tests_to_target is not None:
            if (
                isinstance(self.tests_to_target, bool)
                or not isinstance(self.tests_to_target, int)
                or not 1 <= self.tests_to_target <= self.executed_test_count
            ):
                raise ValueError("tests_to_target geçersiz.")
            if not self.target_reached:
                raise ValueError("Target'a ulaşmayan sonuç tests_to_target taşıyamaz.")
        elif self.target_reached:
            raise ValueError("Target'a ulaşan sonuç tests_to_target taşımalıdır.")
        if not isinstance(self.stop_reason, StrategyStopReason):
            raise TypeError("stop_reason geçersiz.")
        if self.target_reached != (
            self.stop_reason is StrategyStopReason.TARGET_REACHED
        ):
            raise ValueError("Target ve stop reason tutarsız.")
        _validate_finite_non_negative(self.duration_seconds, "duration_seconds")
        for name in (
            "episode_count",
            "q_state_count",
            "actions_evaluated",
            "exploration_selection_count",
            "exploitation_selection_count",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_non_negative_integer(value, name)
        for name in ("cumulative_reward", "final_reward"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} sonlu sayı veya None olmalıdır.")
        if self.strategy is not StrategyName.RL and any(
            getattr(self, name) is not None
            for name in (
                "episode_count",
                "q_state_count",
                "actions_evaluated",
                "exploration_selection_count",
                "exploitation_selection_count",
                "cumulative_reward",
                "final_reward",
                "terminal_reason",
            )
        ):
            raise ValueError("RL-specific metrics deterministic strategy'de bulunamaz.")

    @property
    def selected_count(self) -> int:
        return len(self.selected_candidate_ids)

    @property
    def target_reached(self) -> bool:
        return (
            set(self.target_line_identities) <= set(self.covered_line_identities)
            and set(self.target_branch_identities)
            <= set(self.covered_branch_identities)
        )

    @property
    def covered_line_count(self) -> int:
        return len(self.covered_line_identities)

    @property
    def total_line_count(self) -> int:
        return len(self.executable_line_identities)

    @property
    def line_coverage_percent(self) -> float:
        return self._percentage(self.covered_line_count, self.total_line_count)

    @property
    def covered_branch_count(self) -> int:
        return len(self.covered_branch_identities)

    @property
    def total_branch_count(self) -> int:
        return len(self.executable_branch_identities)

    @property
    def branch_coverage_percent(self) -> float:
        return self._percentage(self.covered_branch_count, self.total_branch_count)

    @property
    def combined_covered_count(self) -> int:
        return self.covered_line_count + self.covered_branch_count

    @property
    def combined_total_count(self) -> int:
        return self.total_line_count + self.total_branch_count

    @property
    def combined_coverage_percent(self) -> float:
        return self._percentage(self.combined_covered_count, self.combined_total_count)

    @staticmethod
    def _percentage(covered: int, total: int) -> float:
        return 100.0 if total == 0 else covered / total * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "seed": self.seed,
            "candidate_pool_count": self.candidate_pool_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_count": self.selected_count,
            "executed_test_count": self.executed_test_count,
            "effective_test_count": self.effective_test_count,
            "zero_contribution_selection_count": (
                self.zero_contribution_selection_count
            ),
            "covered_line_identities": [
                list(value) for value in self.covered_line_identities
            ],
            "covered_line_count": self.covered_line_count,
            "total_line_count": self.total_line_count,
            "line_coverage_percent": self.line_coverage_percent,
            "covered_branch_identities": [
                list(value) for value in self.covered_branch_identities
            ],
            "covered_branch_count": self.covered_branch_count,
            "total_branch_count": self.total_branch_count,
            "branch_coverage_percent": self.branch_coverage_percent,
            "combined_covered_count": self.combined_covered_count,
            "combined_total_count": self.combined_total_count,
            "combined_coverage_percent": self.combined_coverage_percent,
            "target_reached": self.target_reached,
            "tests_to_target": self.tests_to_target,
            "stop_reason": self.stop_reason.value,
            "duration_seconds": self.duration_seconds,
            "coverage_progression": [
                step.to_dict() for step in self.coverage_progression
            ],
            "episode_count": self.episode_count,
            "q_state_count": self.q_state_count,
            "actions_evaluated": self.actions_evaluated,
            "exploration_selection_count": self.exploration_selection_count,
            "exploitation_selection_count": self.exploitation_selection_count,
            "cumulative_reward": self.cumulative_reward,
            "final_reward": self.final_reward,
            "terminal_reason": self.terminal_reason,
        }


@dataclass(frozen=True, slots=True)
class MetricDistribution:
    sample_count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count pozitif tam sayı olmalıdır.")
        for name in (
            "mean",
            "median",
            "minimum",
            "maximum",
            "standard_deviation",
        ):
            _validate_finite_non_negative(getattr(self, name), name)
        if self.minimum > self.maximum:
            raise ValueError("minimum maximum değerini aşamaz.")

    @classmethod
    def from_values(cls, values: tuple[float, ...]) -> MetricDistribution:
        if not values:
            raise ValueError("Metric distribution için örnek gereklidir.")
        normalized = tuple(float(value) for value in values)
        return cls(
            sample_count=len(normalized),
            mean=statistics.fmean(normalized),
            median=statistics.median(normalized),
            minimum=min(normalized),
            maximum=max(normalized),
            standard_deviation=statistics.pstdev(normalized),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True, slots=True)
class RLStrategyAggregate:
    seed_count: int
    selected_tests: MetricDistribution
    effective_tests: MetricDistribution
    combined_coverage: MetricDistribution
    tests_to_target: MetricDistribution | None
    duration_seconds: MetricDistribution

    def __post_init__(self) -> None:
        if self.seed_count != self.selected_tests.sample_count:
            raise ValueError("seed_count selected metric sayısıyla uyuşmuyor.")
        for distribution in (
            self.effective_tests,
            self.combined_coverage,
            self.duration_seconds,
        ):
            if distribution.sample_count != self.seed_count:
                raise ValueError("Aggregate metric seed sayısıyla uyuşmuyor.")
        if (
            self.tests_to_target is not None
            and self.tests_to_target.sample_count > self.seed_count
        ):
            raise ValueError("tests_to_target sample sayısı seed sayısını aşamaz.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_count": self.seed_count,
            "selected_tests": self.selected_tests.to_dict(),
            "effective_tests": self.effective_tests.to_dict(),
            "combined_coverage": self.combined_coverage.to_dict(),
            "tests_to_target": (
                self.tests_to_target.to_dict()
                if self.tests_to_target is not None
                else None
            ),
            "duration_seconds": self.duration_seconds.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StrategyEvaluationReport:
    common_candidate_ids: tuple[str, ...]
    dqm_prefix: StrategyEvaluationResult
    exact_greedy: StrategyEvaluationResult
    rl_runs: tuple[StrategyEvaluationResult, ...]
    rl_aggregate: RLStrategyAggregate
    winner: StrategyWinnerOutcome
    globally_minimal: bool = False

    def __post_init__(self) -> None:
        if not self.common_candidate_ids or len(set(self.common_candidate_ids)) != len(
            self.common_candidate_ids
        ):
            raise ValueError("common_candidate_ids dolu ve benzersiz olmalıdır.")
        results = (self.dqm_prefix, self.exact_greedy, *self.rl_runs)
        if self.dqm_prefix.strategy is not StrategyName.DQM_PREFIX:
            raise ValueError("dqm_prefix strategy uyuşmuyor.")
        if self.exact_greedy.strategy is not StrategyName.EXACT_GREEDY:
            raise ValueError("exact_greedy strategy uyuşmuyor.")
        if not self.rl_runs or any(
            result.strategy is not StrategyName.RL for result in self.rl_runs
        ):
            raise ValueError("rl_runs dolu RL result tuple'ı olmalıdır.")
        if len({result.seed for result in self.rl_runs}) != len(self.rl_runs):
            raise ValueError("RL seed değerleri benzersiz olmalıdır.")
        for result in results:
            if result.candidate_pool_count != len(self.common_candidate_ids):
                raise ValueError("Strategy candidate pool count uyuşmuyor.")
        reference = self.dqm_prefix
        for result in results[1:]:
            if (
                result.executable_line_identities
                != reference.executable_line_identities
                or result.executable_branch_identities
                != reference.executable_branch_identities
                or result.target_line_identities != reference.target_line_identities
                or result.target_branch_identities
                != reference.target_branch_identities
            ):
                raise ValueError("Strategy denominator/target identity uyuşmuyor.")
        if self.rl_aggregate.seed_count != len(self.rl_runs):
            raise ValueError("RL aggregate seed count uyuşmuyor.")
        if not isinstance(self.winner, StrategyWinnerOutcome):
            raise TypeError("winner geçersiz.")
        if self.globally_minimal is not False:
            raise ValueError("Strategy evaluation global minimum iddia edemez.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "common_candidate_ids": list(self.common_candidate_ids),
            "common_candidate_pool_count": len(self.common_candidate_ids),
            "dqm_prefix": self.dqm_prefix.to_dict(),
            "exact_greedy": self.exact_greedy.to_dict(),
            "rl_runs": [result.to_dict() for result in self.rl_runs],
            "rl_aggregate": self.rl_aggregate.to_dict(),
            "winner": self.winner.value,
            "winner_objective": (
                "TARGET_REACHED,SELECTED,EXECUTED,ZERO_CONTRIBUTION,DURATION"
            ),
            "globally_minimal": self.globally_minimal,
        }
