from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from models.strategy_evaluation_result import (
    MetricDistribution,
    RLStrategyAggregate,
    StrategyCoverageStep,
    StrategyEvaluationCandidate,
    StrategyEvaluationPool,
    StrategyEvaluationReport,
    StrategyEvaluationResult,
    StrategyName,
    StrategyStopReason,
    StrategyWinnerOutcome,
)
from rl.action import Action
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_learning_trainer import QLearningTrainer
from rl.q_table import QTable
from rl.state_encoder import StateEncoder
from rl.training_session import TrainingSession
from rl.training_statistics import EpisodeStatistics, TrainingStatistics
from services.project_coverage_service import (
    ProjectExactCoverageGreedySelector,
    ProjectScenarioCoverageSignature,
)


@dataclass(frozen=True, slots=True)
class _TransitionRecord:
    candidate_id: str
    new_lines: tuple[tuple[str, int], ...]
    new_branches: tuple[tuple[str, int, int], ...]
    covered_lines: tuple[tuple[str, int], ...]
    covered_branches: tuple[tuple[str, int, int], ...]
    combined_percentage: float


class _ExactIdentityTransition:
    """Precomputed execution oracle'ını episode-local exact kümeye uygular."""

    def __init__(self, pool: StrategyEvaluationPool) -> None:
        self._pool = pool
        self._covered_lines: set[tuple[str, int]] = set()
        self._covered_branches: set[tuple[str, int, int]] = set()
        self._records: list[_TransitionRecord] = []

    @property
    def records(self) -> tuple[_TransitionRecord, ...]:
        return tuple(self._records)

    def reset(self) -> None:
        self._covered_lines.clear()
        self._covered_branches.clear()
        self._records.clear()

    def __call__(self, state: CoverageState, action: Action) -> CoverageState:
        if action.scenario_index >= len(self._pool.candidates):
            raise RuntimeError("RL action ortak candidate pool dışında.")
        candidate = self._pool.candidates[action.scenario_index]
        previous_lines = set(self._covered_lines)
        previous_branches = set(self._covered_branches)
        self._covered_lines.update(candidate.covered_line_identities)
        self._covered_branches.update(candidate.covered_branch_identities)
        new_lines = tuple(sorted(self._covered_lines - previous_lines))
        new_branches = tuple(sorted(self._covered_branches - previous_branches))
        percentage = self._combined_percentage()
        self._records.append(
            _TransitionRecord(
                candidate_id=candidate.candidate_id,
                new_lines=new_lines,
                new_branches=new_branches,
                covered_lines=tuple(sorted(self._covered_lines)),
                covered_branches=tuple(sorted(self._covered_branches)),
                combined_percentage=percentage,
            )
        )
        missing_line_count = (
            len(self._pool.executable_line_identities) - len(self._covered_lines)
        )
        return CoverageState(
            coverage_percentage=percentage,
            executed_tests=state.executed_tests + 1,
            missing_lines=tuple(range(1, missing_line_count + 1)),
            uncovered_branches=(
                len(self._pool.executable_branch_identities)
                - len(self._covered_branches)
            ),
        )

    def _combined_percentage(self) -> float:
        total = (
            len(self._pool.executable_line_identities)
            + len(self._pool.executable_branch_identities)
        )
        covered = len(self._covered_lines) + len(self._covered_branches)
        return covered / total * 100.0


class StrategyEvaluationService:
    """Aynı execution-derived havuzda DQM, exact greedy ve RL'yi ölçer."""

    _MAX_SEEDS = 100
    _MAX_EPISODES = 1_000
    _MAX_TOTAL_RL_ACTIONS = 1_000_000

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        if not callable(clock):
            raise TypeError("clock callable olmalıdır.")
        self._clock = clock

    @staticmethod
    def pool_from_project_signatures(
        *,
        signatures: tuple[ProjectScenarioCoverageSignature, ...],
        executable_line_identities: tuple[tuple[str, int], ...],
        executable_branch_identities: tuple[tuple[str, int, int], ...],
        target_line_identities: tuple[tuple[str, int], ...],
        target_branch_identities: tuple[tuple[str, int, int], ...],
    ) -> StrategyEvaluationPool:
        """Project coverage oracle çıktısını kayıpsız ortak havuza dönüştürür."""
        if not isinstance(signatures, tuple) or not signatures:
            raise ValueError("signatures dolu tuple olmalıdır.")
        if any(
            not isinstance(signature, ProjectScenarioCoverageSignature)
            for signature in signatures
        ):
            raise TypeError("signatures project coverage signature içermelidir.")
        ordered = tuple(
            sorted(
                signatures,
                key=lambda item: (item.original_order, item.project_test_id),
            )
        )
        return StrategyEvaluationPool(
            candidates=tuple(
                StrategyEvaluationCandidate(
                    candidate_id=signature.project_test_id,
                    dqm_rank=signature.original_order,
                    covered_line_identities=signature.covered_line_identities,
                    covered_branch_identities=signature.covered_branch_identities,
                    execution_success=signature.execution_success,
                )
                for signature in ordered
            ),
            executable_line_identities=tuple(sorted(executable_line_identities)),
            executable_branch_identities=tuple(sorted(executable_branch_identities)),
            target_line_identities=tuple(sorted(target_line_identities)),
            target_branch_identities=tuple(sorted(target_branch_identities)),
        )

    def evaluate(
        self,
        *,
        pool: StrategyEvaluationPool,
        rl_seeds: tuple[int, ...],
        rl_episode_count: int = 10,
        epsilon: float = 0.30,
        epsilon_decay_rate: float | None = 0.95,
        minimum_epsilon: float = 0.05,
        learning_rate: float = 0.5,
        discount_factor: float = 0.9,
    ) -> StrategyEvaluationReport:
        self._validate_pool(pool)
        seeds = self._validate_seeds(rl_seeds)
        self._validate_rl_configuration(
            candidate_count=len(pool.candidates),
            seed_count=len(seeds),
            episode_count=rl_episode_count,
            epsilon=epsilon,
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        dqm = self.evaluate_one(pool=pool, strategy=StrategyName.DQM_PREFIX)
        greedy = self.evaluate_one(pool=pool, strategy=StrategyName.EXACT_GREEDY)
        rl_runs = tuple(
            self.evaluate_one(
                pool=pool,
                strategy=StrategyName.RL,
                seed=seed,
                rl_episode_count=rl_episode_count,
                epsilon=epsilon,
                epsilon_decay_rate=epsilon_decay_rate,
                minimum_epsilon=minimum_epsilon,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            for seed in seeds
        )
        return StrategyEvaluationReport(
            common_candidate_ids=pool.candidate_ids,
            dqm_prefix=dqm,
            exact_greedy=greedy,
            rl_runs=rl_runs,
            rl_aggregate=self._aggregate(rl_runs),
            winner=self.determine_winner(
                dqm=dqm,
                greedy=greedy,
                rl_runs=rl_runs,
            ),
        )

    @staticmethod
    def write_report(
        report: StrategyEvaluationReport,
        *,
        output_root: str | Path,
    ) -> Path:
        """Güvenli aggregate evaluation özetini atomik JSON olarak yazar."""
        if not isinstance(report, StrategyEvaluationReport):
            raise TypeError("report StrategyEvaluationReport olmalıdır.")
        root = Path(output_root).resolve()
        path = (root / "strategy_evaluation" / "strategy_evaluation_report.json").resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("Strategy evaluation report output root dışında.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    report.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def determine_winner(
        cls,
        *,
        dqm: StrategyEvaluationResult,
        greedy: StrategyEvaluationResult,
        rl_runs: tuple[StrategyEvaluationResult, ...],
    ) -> StrategyWinnerOutcome:
        """Önceden ilan edilen lexicographic objective'i saf biçimde uygular."""
        if dqm.strategy is not StrategyName.DQM_PREFIX:
            raise ValueError("dqm sonucu DQM_PREFIX olmalıdır.")
        if greedy.strategy is not StrategyName.EXACT_GREEDY:
            raise ValueError("greedy sonucu EXACT_GREEDY olmalıdır.")
        if not rl_runs or any(
            result.strategy is not StrategyName.RL for result in rl_runs
        ):
            raise ValueError("rl_runs dolu RL result tuple'ı olmalıdır.")
        return cls._winner(dqm=dqm, greedy=greedy, rl_runs=rl_runs)

    def evaluate_one(
        self,
        *,
        pool: StrategyEvaluationPool,
        strategy: StrategyName,
        seed: int | None = None,
        rl_episode_count: int = 10,
        epsilon: float = 0.30,
        epsilon_decay_rate: float | None = 0.95,
        minimum_epsilon: float = 0.05,
        learning_rate: float = 0.5,
        discount_factor: float = 0.9,
    ) -> StrategyEvaluationResult:
        self._validate_pool(pool)
        if not isinstance(strategy, StrategyName):
            raise TypeError("strategy StrategyName olmalıdır.")
        if strategy is StrategyName.DQM_PREFIX:
            if seed is not None:
                raise ValueError("DQM_PREFIX seed kabul etmez.")
            return self._evaluate_ordered(
                pool=pool,
                strategy=strategy,
                candidates=pool.candidates,
            )
        if strategy is StrategyName.EXACT_GREEDY:
            if seed is not None:
                raise ValueError("EXACT_GREEDY seed kabul etmez.")
            return self._evaluate_ordered(
                pool=pool,
                strategy=strategy,
                candidates=self._greedy_order(pool),
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("RL evaluation için seed tam sayı olmalıdır.")
        self._validate_rl_configuration(
            candidate_count=len(pool.candidates),
            seed_count=1,
            episode_count=rl_episode_count,
            epsilon=epsilon,
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        return self._evaluate_rl(
            pool=pool,
            seed=seed,
            episode_count=rl_episode_count,
            epsilon=epsilon,
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def _evaluate_ordered(
        self,
        *,
        pool: StrategyEvaluationPool,
        strategy: StrategyName,
        candidates: tuple[StrategyEvaluationCandidate, ...],
    ) -> StrategyEvaluationResult:
        started = self._clock()
        covered_lines: set[tuple[str, int]] = set()
        covered_branches: set[tuple[str, int, int]] = set()
        progression: list[StrategyCoverageStep] = []
        for candidate in candidates:
            self._append_step(
                pool=pool,
                candidate=candidate,
                covered_lines=covered_lines,
                covered_branches=covered_branches,
                progression=progression,
            )
            if self._target_reached(pool, covered_lines, covered_branches):
                break
        target_reached = self._target_reached(
            pool,
            covered_lines,
            covered_branches,
        )
        reason = (
            StrategyStopReason.TARGET_REACHED
            if target_reached
            else StrategyStopReason.NO_FURTHER_CONTRIBUTION
            if strategy is StrategyName.EXACT_GREEDY
            else StrategyStopReason.CANDIDATES_EXHAUSTED
        )
        return self._build_result(
            pool=pool,
            strategy=strategy,
            seed=None,
            progression=tuple(progression),
            stop_reason=reason,
            duration=max(0.0, self._clock() - started),
        )

    def _evaluate_rl(
        self,
        *,
        pool: StrategyEvaluationPool,
        seed: int,
        episode_count: int,
        epsilon: float,
        epsilon_decay_rate: float | None,
        minimum_epsilon: float,
        learning_rate: float,
        discount_factor: float,
    ) -> StrategyEvaluationResult:
        started = self._clock()
        transition = _ExactIdentityTransition(pool)
        actions = tuple(Action(index) for index in range(len(pool.candidates)))
        environment = CoverageEnvironment(
            initial_state=self._initial_state(pool),
            actions=actions,
            transition_function=transition,
            episode_reset_callback=transition.reset,
            target_coverage_percentage=self._target_percentage(pool),
        )
        q_table = QTable()
        policy = EpsilonGreedyPolicy(
            epsilon=epsilon,
            random_generator=random.Random(seed),
        )
        agent = QLearningAgent(
            q_table=q_table,
            policy=policy,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        trainer = QLearningTrainer(
            agent=agent,
            state_encoder=StateEncoder(
                coverage_bucket_size=10.0,
                missing_lines_bucket_size=2,
                uncovered_branches_bucket_size=1,
                executed_tests_bucket_size=1,
            ),
        )
        statistics = TrainingStatistics()
        session = TrainingSession(trainer=trainer, statistics=statistics)
        episode_results: list[StrategyEvaluationResult] = []

        def completed(episode: EpisodeStatistics) -> None:
            records = transition.records
            progression = tuple(
                StrategyCoverageStep(
                    step_number=index,
                    candidate_id=record.candidate_id,
                    new_line_identities=record.new_lines,
                    new_branch_identities=record.new_branches,
                    covered_line_identities=record.covered_lines,
                    covered_branch_identities=record.covered_branches,
                    combined_coverage_percent=record.combined_percentage,
                )
                for index, record in enumerate(records, start=1)
            )
            reached = bool(progression) and self._target_reached(
                pool,
                set(progression[-1].covered_line_identities),
                set(progression[-1].covered_branch_identities),
            )
            episode_results.append(
                self._build_result(
                    pool=pool,
                    strategy=StrategyName.RL,
                    seed=seed,
                    progression=progression,
                    stop_reason=(
                        StrategyStopReason.TARGET_REACHED
                        if reached
                        else StrategyStopReason.CANDIDATES_EXHAUSTED
                    ),
                    duration=float(episode.duration_seconds or 0.0),
                    episode_count=1,
                    actions_evaluated=len(progression),
                    cumulative_reward=episode.total_reward,
                    final_reward=episode.total_reward,
                    terminal_reason=episode.done_reason,
                )
            )

        session_result = session.run(
            environment=environment,
            episode_count=episode_count,
            epsilon_decay_rate=epsilon_decay_rate,
            minimum_epsilon=minimum_epsilon,
            episode_completed_callback=completed,
        )
        if len(episode_results) != session_result.completed_episode_count:
            raise RuntimeError("RL episode evaluation trace sayısı tutarsız.")
        best_index, best = min(
            enumerate(episode_results, start=1),
            key=lambda item: (
                0 if item[1].target_reached else 1,
                item[1].selected_count,
                item[1].executed_test_count,
                item[1].zero_contribution_selection_count,
                -float(item[1].final_reward or 0.0),
                item[0],
            ),
        )
        del best_index
        return replace(
            best,
            duration_seconds=max(0.0, self._clock() - started),
            episode_count=episode_count,
            q_state_count=len(q_table),
            actions_evaluated=sum(
                result.executed_test_count for result in episode_results
            ),
            exploration_selection_count=policy.exploration_selection_count,
            exploitation_selection_count=policy.exploitation_selection_count,
            cumulative_reward=sum(
                float(result.cumulative_reward or 0.0)
                for result in episode_results
            ),
            final_reward=best.final_reward,
            terminal_reason=best.terminal_reason,
        )

    @staticmethod
    def _greedy_order(
        pool: StrategyEvaluationPool,
    ) -> tuple[StrategyEvaluationCandidate, ...]:
        signatures = tuple(
            ProjectScenarioCoverageSignature(
                project_test_id=candidate.candidate_id,
                original_order=candidate.dqm_rank,
                covered_line_identities=candidate.covered_line_identities,
                covered_branch_identities=candidate.covered_branch_identities,
                execution_success=candidate.execution_success,
            )
            for candidate in pool.candidates
        )
        selection = ProjectExactCoverageGreedySelector.select(
            signatures=signatures,
            target_lines=pool.target_line_identities,
            target_branches=pool.target_branch_identities,
        )
        by_identifier = {
            candidate.candidate_id: candidate for candidate in pool.candidates
        }
        return tuple(
            by_identifier[identifier]
            for identifier in selection.greedy_selection_order_ids
        )

    def _append_step(
        self,
        *,
        pool: StrategyEvaluationPool,
        candidate: StrategyEvaluationCandidate,
        covered_lines: set[tuple[str, int]],
        covered_branches: set[tuple[str, int, int]],
        progression: list[StrategyCoverageStep],
    ) -> None:
        previous_lines = set(covered_lines)
        previous_branches = set(covered_branches)
        covered_lines.update(candidate.covered_line_identities)
        covered_branches.update(candidate.covered_branch_identities)
        progression.append(
            StrategyCoverageStep(
                step_number=len(progression) + 1,
                candidate_id=candidate.candidate_id,
                new_line_identities=tuple(sorted(covered_lines - previous_lines)),
                new_branch_identities=tuple(
                    sorted(covered_branches - previous_branches)
                ),
                covered_line_identities=tuple(sorted(covered_lines)),
                covered_branch_identities=tuple(sorted(covered_branches)),
                combined_coverage_percent=self._combined_percentage(
                    pool,
                    len(covered_lines),
                    len(covered_branches),
                ),
            )
        )

    @staticmethod
    def _build_result(
        *,
        pool: StrategyEvaluationPool,
        strategy: StrategyName,
        seed: int | None,
        progression: tuple[StrategyCoverageStep, ...],
        stop_reason: StrategyStopReason,
        duration: float,
        episode_count: int | None = None,
        q_state_count: int | None = None,
        actions_evaluated: int | None = None,
        exploration_selection_count: int | None = None,
        exploitation_selection_count: int | None = None,
        cumulative_reward: float | None = None,
        final_reward: float | None = None,
        terminal_reason: str | None = None,
    ) -> StrategyEvaluationResult:
        covered_lines = (
            progression[-1].covered_line_identities if progression else ()
        )
        covered_branches = (
            progression[-1].covered_branch_identities if progression else ()
        )
        target_reached = (
            set(pool.target_line_identities) <= set(covered_lines)
            and set(pool.target_branch_identities) <= set(covered_branches)
        )
        effective = sum(step.exact_new_identity_count > 0 for step in progression)
        return StrategyEvaluationResult(
            strategy=strategy,
            seed=seed,
            candidate_pool_count=len(pool.candidates),
            selected_candidate_ids=tuple(step.candidate_id for step in progression),
            executed_test_count=len(progression),
            effective_test_count=effective,
            zero_contribution_selection_count=len(progression) - effective,
            executable_line_identities=pool.executable_line_identities,
            executable_branch_identities=pool.executable_branch_identities,
            target_line_identities=pool.target_line_identities,
            target_branch_identities=pool.target_branch_identities,
            covered_line_identities=covered_lines,
            covered_branch_identities=covered_branches,
            coverage_progression=progression,
            tests_to_target=len(progression) if target_reached else None,
            stop_reason=stop_reason,
            duration_seconds=duration,
            episode_count=episode_count,
            q_state_count=q_state_count,
            actions_evaluated=actions_evaluated,
            exploration_selection_count=exploration_selection_count,
            exploitation_selection_count=exploitation_selection_count,
            cumulative_reward=cumulative_reward,
            final_reward=final_reward,
            terminal_reason=terminal_reason,
        )

    @staticmethod
    def _initial_state(pool: StrategyEvaluationPool) -> CoverageState:
        return CoverageState(
            coverage_percentage=0.0,
            executed_tests=0,
            missing_lines=tuple(range(1, len(pool.executable_line_identities) + 1)),
            uncovered_branches=len(pool.executable_branch_identities),
        )

    @staticmethod
    def _target_percentage(pool: StrategyEvaluationPool) -> float:
        target = len(pool.target_line_identities) + len(pool.target_branch_identities)
        total = len(pool.executable_line_identities) + len(
            pool.executable_branch_identities
        )
        return target / total * 100.0

    @staticmethod
    def _combined_percentage(
        pool: StrategyEvaluationPool,
        covered_lines: int,
        covered_branches: int,
    ) -> float:
        total = len(pool.executable_line_identities) + len(
            pool.executable_branch_identities
        )
        return (covered_lines + covered_branches) / total * 100.0

    @staticmethod
    def _target_reached(
        pool: StrategyEvaluationPool,
        covered_lines: set[tuple[str, int]],
        covered_branches: set[tuple[str, int, int]],
    ) -> bool:
        return set(pool.target_line_identities) <= covered_lines and set(
            pool.target_branch_identities
        ) <= covered_branches

    @staticmethod
    def _aggregate(
        rl_runs: tuple[StrategyEvaluationResult, ...],
    ) -> RLStrategyAggregate:
        reached = tuple(
            float(result.tests_to_target)
            for result in rl_runs
            if result.tests_to_target is not None
        )
        return RLStrategyAggregate(
            seed_count=len(rl_runs),
            selected_tests=MetricDistribution.from_values(
                tuple(float(result.selected_count) for result in rl_runs)
            ),
            effective_tests=MetricDistribution.from_values(
                tuple(float(result.effective_test_count) for result in rl_runs)
            ),
            combined_coverage=MetricDistribution.from_values(
                tuple(result.combined_coverage_percent for result in rl_runs)
            ),
            tests_to_target=(
                MetricDistribution.from_values(reached) if reached else None
            ),
            duration_seconds=MetricDistribution.from_values(
                tuple(result.duration_seconds for result in rl_runs)
            ),
        )

    @classmethod
    def _winner(
        cls,
        *,
        dqm: StrategyEvaluationResult,
        greedy: StrategyEvaluationResult,
        rl_runs: tuple[StrategyEvaluationResult, ...],
    ) -> StrategyWinnerOutcome:
        baseline = min((dqm, greedy), key=cls._objective_key)
        outcomes: list[StrategyWinnerOutcome] = []
        baseline_key = cls._objective_key(baseline)
        for rl_result in rl_runs:
            rl_key = cls._objective_key(rl_result)
            outcomes.append(
                StrategyWinnerOutcome.RL_WINS
                if rl_key < baseline_key
                else StrategyWinnerOutcome.BASELINE_WINS
                if baseline_key < rl_key
                else StrategyWinnerOutcome.TIE
            )
        unique = set(outcomes)
        return outcomes[0] if len(unique) == 1 else StrategyWinnerOutcome.MIXED

    @staticmethod
    def _objective_key(result: StrategyEvaluationResult) -> tuple[object, ...]:
        return (
            0 if result.target_reached else 1,
            result.selected_count,
            result.executed_test_count,
            result.zero_contribution_selection_count,
            result.duration_seconds,
        )

    @staticmethod
    def _validate_pool(pool: StrategyEvaluationPool) -> None:
        if not isinstance(pool, StrategyEvaluationPool):
            raise TypeError("pool StrategyEvaluationPool olmalıdır.")

    @classmethod
    def _validate_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        if not isinstance(seeds, tuple) or not seeds:
            raise ValueError("rl_seeds dolu tuple olmalıdır.")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("rl_seeds yalnız tam sayı içermelidir.")
        if len(set(seeds)) != len(seeds):
            raise ValueError("rl_seeds benzersiz olmalıdır.")
        if len(seeds) > cls._MAX_SEEDS:
            raise ValueError("rl_seeds güvenli üst sınırı aşıyor.")
        return seeds

    @classmethod
    def _validate_rl_configuration(
        cls,
        *,
        candidate_count: int,
        seed_count: int,
        episode_count: int,
        epsilon: float,
        epsilon_decay_rate: float | None,
        minimum_epsilon: float,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
            or not 1 <= episode_count <= cls._MAX_EPISODES
        ):
            raise ValueError("rl_episode_count güvenli aralıkta olmalıdır.")
        if candidate_count * seed_count * episode_count > cls._MAX_TOTAL_RL_ACTIONS:
            raise ValueError("RL evaluation action budget güvenli üst sınırı aşıyor.")
        for name, value in (
            ("epsilon", epsilon),
            ("minimum_epsilon", minimum_epsilon),
            ("learning_rate", learning_rate),
            ("discount_factor", discount_factor),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} 0 ile 1 arasında sonlu sayı olmalıdır.")
        if epsilon_decay_rate is not None and (
            isinstance(epsilon_decay_rate, bool)
            or not isinstance(epsilon_decay_rate, (int, float))
            or not math.isfinite(float(epsilon_decay_rate))
            or not 0.0 <= float(epsilon_decay_rate) <= 1.0
        ):
            raise ValueError("epsilon_decay_rate 0 ile 1 arasında veya None olmalıdır.")
        if epsilon_decay_rate is not None and minimum_epsilon > epsilon:
            raise ValueError("minimum_epsilon başlangıç epsilon değerini aşamaz.")
