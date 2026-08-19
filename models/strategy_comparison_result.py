from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class StrategyComparisonStatus(str, Enum):
    COMPLETED = "COMPLETED"
    TARGET_NOT_REACHED = "TARGET_NOT_REACHED"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    TIMED_OUT = "TIMED_OUT"
    PARTIAL = "PARTIAL"


class StrategyWinner(str, Enum):
    RL = "RL"
    GREEDY = "GREEDY"
    TIE = "TIE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True, slots=True)
class EpisodeSelectionTrace:
    episode_number: int
    ordered_scenario_ids: tuple[str, ...]
    executed_action_count: int
    unique_scenario_count: int
    duplicate_selection_count: int
    final_reported_coverage: float
    total_reward: float
    done_reason: str | None
    duration_seconds: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_number, bool)
            or not isinstance(self.episode_number, int)
            or self.episode_number < 1
        ):
            raise ValueError("episode_number pozitif tam sayı olmalıdır.")
        if not isinstance(self.ordered_scenario_ids, tuple) or any(
            not isinstance(value, str) or not value
            for value in self.ordered_scenario_ids
        ):
            raise TypeError("ordered_scenario_ids dolu string tuple'ı olmalıdır.")
        if self.executed_action_count != len(self.ordered_scenario_ids):
            raise ValueError("executed_action_count ordered scenario sayısıyla uyuşmuyor.")
        unique_count = len(set(self.ordered_scenario_ids))
        if self.unique_scenario_count != unique_count:
            raise ValueError("unique_scenario_count tutarsız.")
        if self.duplicate_selection_count != self.executed_action_count - unique_count:
            raise ValueError("duplicate_selection_count tutarsız.")
        self._validate_percentage(self.final_reported_coverage)
        self._validate_finite(self.total_reward, "total_reward")
        if self.duration_seconds is not None:
            self._validate_finite(self.duration_seconds, "duration_seconds")
            if self.duration_seconds < 0.0:
                raise ValueError("duration_seconds negatif olamaz.")
        if self.done_reason is not None and (
            not isinstance(self.done_reason, str) or not self.done_reason
        ):
            raise TypeError("done_reason string veya None olmalıdır.")

    @staticmethod
    def _validate_percentage(value: float) -> None:
        EpisodeSelectionTrace._validate_finite(value, "final_reported_coverage")
        if not 0.0 <= float(value) <= 100.0:
            raise ValueError("final_reported_coverage 0 ile 100 arasında olmalıdır.")

    @staticmethod
    def _validate_finite(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} sayısal olmalıdır.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} sonlu olmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_number": self.episode_number,
            "ordered_scenario_ids": list(self.ordered_scenario_ids),
            "executed_action_count": self.executed_action_count,
            "unique_scenario_count": self.unique_scenario_count,
            "duplicate_selection_count": self.duplicate_selection_count,
            "final_reported_coverage": self.final_reported_coverage,
            "total_reward": self.total_reward,
            "done_reason": self.done_reason,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class StrategyComparisonResult:
    source_file: Path
    function_name: str
    status: StrategyComparisonStatus
    comparable: bool
    non_comparable_reason: str | None
    scenario_pool_count: int
    target_line_identities: tuple[int, ...]
    target_line_percentage: float | None
    target_branch_identities: tuple[tuple[int, int], ...]
    target_branch_percentage: float | None
    greedy_algorithm: str
    greedy_selected_scenario_ids: tuple[str, ...]
    greedy_coverage_preserved: bool
    greedy_duration_seconds: float | None
    requested_rl_episode_count: int
    completed_rl_episode_count: int
    rl_hyperparameters: tuple[tuple[str, object], ...]
    random_seed: int | None
    episode_traces: tuple[EpisodeSelectionTrace, ...]
    exact_target_reaching_episode_count: int
    best_rl_episode_number: int | None
    best_rl_ordered_scenario_ids: tuple[str, ...]
    best_rl_executed_test_count: int | None
    best_rl_unique_scenario_count: int | None
    best_rl_duplicate_count: int | None
    best_rl_total_reward: float | None
    rl_coverage_preserved: bool
    rl_duration_seconds: float | None
    winner: StrategyWinner
    coverage_equality_verified: bool
    artifact_paths: tuple[Path, ...] = ()
    failure_category: str | None = None
    failure_message: str | None = None
    greedy_verified_line_identities: tuple[int, ...] = ()
    greedy_verified_branch_identities: tuple[tuple[int, int], ...] = ()
    rl_verified_line_identities: tuple[int, ...] = ()
    rl_verified_branch_identities: tuple[tuple[int, int], ...] = ()
    globally_minimal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, Path):
            raise TypeError("source_file Path olmalıdır.")
        if not self.function_name or not self.function_name.isidentifier():
            raise ValueError("function_name geçerli identifier olmalıdır.")
        if not isinstance(self.status, StrategyComparisonStatus):
            raise TypeError("status geçersiz.")
        if not isinstance(self.comparable, bool):
            raise TypeError("comparable bool olmalıdır.")
        if not isinstance(self.winner, StrategyWinner):
            raise TypeError("winner geçersiz.")
        if self.globally_minimal is not False:
            raise ValueError("Comparison global minimum iddia edemez.")
        for name in (
            "scenario_pool_count",
            "requested_rl_episode_count",
            "completed_rl_episode_count",
            "exact_target_reaching_episode_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} negatif olmayan tam sayı olmalıdır.")
        if self.completed_rl_episode_count > self.requested_rl_episode_count:
            raise ValueError("Completed episode sayısı requested sayıyı aşamaz.")
        if len(self.episode_traces) != self.completed_rl_episode_count:
            raise ValueError("Episode trace sayısı completed episode sayısıyla uyuşmuyor.")
        if self.exact_target_reaching_episode_count > len(self.episode_traces):
            raise ValueError("Exact-target episode sayısı trace sayısını aşamaz.")
        if self.comparable and self.winner is StrategyWinner.NOT_COMPARABLE:
            raise ValueError("Comparable sonuç NOT_COMPARABLE winner taşıyamaz.")
        if not self.comparable and self.winner is not StrategyWinner.NOT_COMPARABLE:
            raise ValueError("Non-comparable sonuç winner taşıyamaz.")
        if self.comparable and self.non_comparable_reason is not None:
            raise ValueError("Comparable sonuç non_comparable_reason taşıyamaz.")
        if not self.comparable and not self.non_comparable_reason:
            raise ValueError("Non-comparable sonuç reason taşımalıdır.")

    @property
    def target_line_count(self) -> int:
        return len(self.target_line_identities)

    @property
    def target_branch_count(self) -> int:
        return len(self.target_branch_identities)

    @property
    def greedy_selected_count(self) -> int:
        return len(self.greedy_selected_scenario_ids)

    @property
    def rl_reduction_percentage(self) -> float | None:
        return self._reduction(self.best_rl_executed_test_count)

    @property
    def greedy_reduction_percentage(self) -> float | None:
        return self._reduction(self.greedy_selected_count)

    def _reduction(self, count: int | None) -> float | None:
        if count is None or self.scenario_pool_count == 0:
            return None
        return (self.scenario_pool_count - count) / self.scenario_pool_count * 100.0

    @property
    def rl_count_minus_greedy_count(self) -> int | None:
        if self.best_rl_executed_test_count is None:
            return None
        return self.best_rl_executed_test_count - self.greedy_selected_count

    @property
    def test_count_difference(self) -> int | None:
        difference = self.rl_count_minus_greedy_count
        return abs(difference) if difference is not None else None

    @staticmethod
    def determine_winner(
        *,
        greedy_coverage_preserved: bool,
        rl_coverage_preserved: bool,
        greedy_selected_count: int,
        rl_selected_count: int,
    ) -> StrategyWinner:
        if greedy_coverage_preserved and rl_coverage_preserved:
            if rl_selected_count < greedy_selected_count:
                return StrategyWinner.RL
            if greedy_selected_count < rl_selected_count:
                return StrategyWinner.GREEDY
            return StrategyWinner.TIE
        if rl_coverage_preserved:
            return StrategyWinner.RL
        if greedy_coverage_preserved:
            return StrategyWinner.GREEDY
        return StrategyWinner.NOT_COMPARABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "source_file": str(self.source_file),
            "function_name": self.function_name,
            "comparison_status": self.status.value,
            "comparable": self.comparable,
            "non_comparable_reason": self.non_comparable_reason,
            "scenario_pool_count": self.scenario_pool_count,
            "target_line_identities": list(self.target_line_identities),
            "target_line_count": self.target_line_count,
            "target_line_percentage": self.target_line_percentage,
            "target_branch_identities": [list(value) for value in self.target_branch_identities],
            "target_branch_count": self.target_branch_count,
            "target_branch_percentage": self.target_branch_percentage,
            "greedy_algorithm": self.greedy_algorithm,
            "greedy_selected_scenario_ids": list(self.greedy_selected_scenario_ids),
            "greedy_selected_count": self.greedy_selected_count,
            "greedy_coverage_preserved": self.greedy_coverage_preserved,
            "greedy_verified_line_identities": list(self.greedy_verified_line_identities),
            "greedy_verified_branch_identities": [
                list(value) for value in self.greedy_verified_branch_identities
            ],
            "greedy_duration_seconds": self.greedy_duration_seconds,
            "requested_rl_episode_count": self.requested_rl_episode_count,
            "completed_rl_episode_count": self.completed_rl_episode_count,
            "rl_hyperparameters": dict(self.rl_hyperparameters),
            "random_seed": self.random_seed,
            "episode_traces": [trace.to_dict() for trace in self.episode_traces],
            "exact_target_reaching_episode_count": self.exact_target_reaching_episode_count,
            "best_rl_episode_number": self.best_rl_episode_number,
            "best_rl_ordered_scenario_ids": list(self.best_rl_ordered_scenario_ids),
            "best_rl_executed_test_count": self.best_rl_executed_test_count,
            "best_rl_unique_scenario_count": self.best_rl_unique_scenario_count,
            "best_rl_duplicate_count": self.best_rl_duplicate_count,
            "best_rl_total_reward": self.best_rl_total_reward,
            "rl_coverage_preserved": self.rl_coverage_preserved,
            "rl_verified_line_identities": list(self.rl_verified_line_identities),
            "rl_verified_branch_identities": [
                list(value) for value in self.rl_verified_branch_identities
            ],
            "rl_duration_seconds": self.rl_duration_seconds,
            "winner": self.winner.value,
            "test_count_difference": self.test_count_difference,
            "rl_reduction_percentage": self.rl_reduction_percentage,
            "greedy_reduction_percentage": self.greedy_reduction_percentage,
            "rl_count_minus_greedy_count": self.rl_count_minus_greedy_count,
            "coverage_equality_verified": self.coverage_equality_verified,
            "globally_minimal": self.globally_minimal,
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
        }

    def format_summary(self) -> str:
        return "\n".join(
            (
                "RL / GREEDY STRATEGY COMPARISON",
                f"Validated scenario pool       : {self.scenario_pool_count}",
                f"Greedy test count             : {self.greedy_selected_count}",
                f"Greedy coverage preserved     : {self.greedy_coverage_preserved}",
                f"RL requested/completed        : {self.requested_rl_episode_count}/{self.completed_rl_episode_count}",
                f"Exact-target RL episodes      : {self.exact_target_reaching_episode_count}",
                f"Best RL episode               : {self.best_rl_episode_number}",
                f"RL executed/unique tests      : {self.best_rl_executed_test_count}/{self.best_rl_unique_scenario_count}",
                f"RL coverage preserved         : {self.rl_coverage_preserved}",
                f"Winner                        : {self.winner.value}",
                f"Comparable                    : {self.comparable}",
                "Global minimum guarantee       : Hayır",
            )
        )
