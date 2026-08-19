from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from generator.scenario_generator import Scenario
from models.coverage_result import FunctionCoverageResult
from models.scenario_minimization_result import ScenarioMinimizationResult
from models.strategy_comparison_result import (
    EpisodeSelectionTrace,
    StrategyComparisonResult,
    StrategyComparisonStatus,
    StrategyWinner,
)
from rl.training_session import TrainingSessionResult
from services.coverage_service import CoverageExecutionTimeoutError
from services.scenario_suite_coverage_service import ScenarioSuiteCoverageService


class ExactRLEpisodeSelector:
    @staticmethod
    def select_best(
        traces: tuple[EpisodeSelectionTrace, ...],
    ) -> EpisodeSelectionTrace:
        if not traces:
            raise ValueError("Exact target'a ulaşan episode bulunamadı.")
        return min(
            traces,
            key=lambda trace: (
                trace.executed_action_count,
                trace.unique_scenario_count,
                -trace.total_reward,
                trace.episode_number,
            ),
        )


class StrategyComparisonService:
    def __init__(
        self,
        *,
        suite_service: ScenarioSuiteCoverageService | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._suite_service = suite_service or ScenarioSuiteCoverageService()
        self._clock = clock

    def write_pending(
        self,
        *,
        source_file: str | Path,
        function_name: str,
        scenarios: tuple[Scenario, ...],
        full_pool_coverage: FunctionCoverageResult,
        greedy_result: ScenarioMinimizationResult,
        requested_rl_episode_count: int,
        output_root: str | Path,
        rl_hyperparameters: tuple[tuple[str, object], ...],
        random_seed: int | None,
        session_result: TrainingSessionResult | None = None,
    ) -> StrategyComparisonResult:
        source = Path(source_file).resolve()
        root = Path(output_root).resolve()
        work_directory = (root / "strategy_comparison").resolve()
        if not work_directory.is_relative_to(root):
            raise RuntimeError("Strategy comparison output yolu root dışına çıktı.")
        if not function_name.isidentifier():
            raise ValueError("function_name geçerli identifier olmalıdır.")
        pending_session = session_result or TrainingSessionResult(
            episodes=(),
            requested_episode_count=requested_rl_episode_count,
            completed_episode_count=0,
        )
        traces = self._build_traces(pending_session, scenarios)
        report_path = work_directory / "strategy_comparison_report.json"
        result = self._result(
            source=source,
            function_name=function_name,
            status=StrategyComparisonStatus.PARTIAL,
            comparable=False,
            reason="RL_TRAINING_PENDING",
            scenarios=scenarios,
            full_pool=full_pool_coverage,
            greedy=greedy_result,
            session=pending_session,
            hyperparameters=rl_hyperparameters,
            random_seed=random_seed,
            traces=traces,
            winner=StrategyWinner.NOT_COMPARABLE,
            equality=False,
            artifacts=(report_path,),
            failure_category="RL_TRAINING_PENDING",
            failure_message="RL episode sonuçları henüz tamamlanmadı.",
        )
        return self._write_report(result, report_path)

    def compare(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        function_start_line: int,
        function_end_line: int,
        scenarios: tuple[Scenario, ...],
        full_pool_coverage: FunctionCoverageResult,
        greedy_result: ScenarioMinimizationResult,
        session_result: TrainingSessionResult,
        output_root: str | Path,
        timeout_seconds: float,
        comparison_timeout_seconds: float | None,
        rl_hyperparameters: tuple[tuple[str, object], ...],
        random_seed: int | None,
    ) -> StrategyComparisonResult:
        started = self._clock()
        source = Path(source_file).resolve()
        root = Path(output_root).resolve()
        work_directory = (root / "strategy_comparison").resolve()
        if not work_directory.is_relative_to(root):
            raise RuntimeError("Strategy comparison output yolu root dışına çıktı.")
        report_path = work_directory / "strategy_comparison_report.json"
        final_test_path = work_directory / f"test_{function_name}_rl_selected.py"
        self._validate_inputs(
            function_name=function_name,
            scenarios=scenarios,
            full_pool_coverage=full_pool_coverage,
            timeout_seconds=timeout_seconds,
            comparison_timeout_seconds=comparison_timeout_seconds,
        )
        traces = self._build_traces(session_result, scenarios)
        target_lines = tuple(full_pool_coverage.covered_lines)
        target_branches = tuple(full_pool_coverage.covered_branches or ())
        target_matches_greedy = (
            greedy_result.coverage_preserved
            and greedy_result.target_line_identities == target_lines
            and greedy_result.target_branch_identities == target_branches
            and greedy_result.final_verified_line_identities == target_lines
            and greedy_result.final_verified_branch_identities == target_branches
        )
        initial = self._result(
            source=source,
            function_name=function_name,
            status=StrategyComparisonStatus.PARTIAL,
            comparable=False,
            reason="RL_SUITE_VERIFICATION_PENDING",
            scenarios=scenarios,
            full_pool=full_pool_coverage,
            greedy=greedy_result,
            session=session_result,
            hyperparameters=rl_hyperparameters,
            random_seed=random_seed,
            traces=traces,
            rl_duration=self._trace_duration(traces),
            winner=StrategyWinner.NOT_COMPARABLE,
            equality=False,
            artifacts=(report_path,),
        )
        self._write_report(initial, report_path)

        candidates = tuple(
            trace
            for trace in traces
            if math.isclose(
                trace.final_reported_coverage,
                full_pool_coverage.line_coverage_percent,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        verification_cache: dict[tuple[str, ...], FunctionCoverageResult] = {}
        exact_traces: list[EpisodeSelectionTrace] = []
        coverage_by_episode: dict[int, FunctionCoverageResult] = {}
        try:
            for trace in candidates:
                if trace.duplicate_selection_count:
                    continue
                suite_key = trace.ordered_scenario_ids
                coverage = verification_cache.get(suite_key)
                if coverage is None:
                    selected = self._select_scenarios(scenarios, suite_key)
                    coverage = self._measure(
                        source=source,
                        module_path=module_path,
                        function_name=function_name,
                        start_line=function_start_line,
                        end_line=function_end_line,
                        scenarios=selected,
                        output_directory=(
                            work_directory
                            / "verification"
                            / f"episode_{trace.episode_number:04d}"
                        ),
                        test_file_name=f"test_{function_name}_rl_candidate.py",
                        timeout=self._effective_timeout(
                            started,
                            timeout_seconds,
                            comparison_timeout_seconds,
                        ),
                    )
                    verification_cache[suite_key] = coverage
                coverage_by_episode[trace.episode_number] = coverage
                if self._preserves_exact(coverage, target_lines, target_branches):
                    exact_traces.append(trace)
        except CoverageExecutionTimeoutError as error:
            result = self._result(
                source=source,
                function_name=function_name,
                status=StrategyComparisonStatus.TIMED_OUT,
                comparable=False,
                reason="COMPARISON_TIMEOUT",
                scenarios=scenarios,
                full_pool=full_pool_coverage,
                greedy=greedy_result,
                session=session_result,
                hyperparameters=rl_hyperparameters,
                random_seed=random_seed,
                traces=traces,
                exact_count=len(exact_traces),
                rl_duration=self._trace_duration(traces),
                winner=StrategyWinner.NOT_COMPARABLE,
                equality=False,
                artifacts=(report_path,),
                failure_category="RL_VERIFICATION_TIMEOUT",
                failure_message=str(error),
            )
            return self._write_report(result, report_path)

        best = (
            ExactRLEpisodeSelector.select_best(tuple(exact_traces))
            if exact_traces
            else None
        )
        rl_preserved = best is not None
        if best is not None:
            verified = coverage_by_episode[best.episode_number]
            work_directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(verified.test_file, final_test_path)

        winner = StrategyComparisonResult.determine_winner(
            greedy_coverage_preserved=target_matches_greedy,
            rl_coverage_preserved=rl_preserved,
            greedy_selected_count=len(greedy_result.final_selected_scenario_ids),
            rl_selected_count=(best.executed_action_count if best else 0),
        )
        comparable = winner is not StrategyWinner.NOT_COMPARABLE
        status = (
            StrategyComparisonStatus.COMPLETED
            if rl_preserved
            else StrategyComparisonStatus.TARGET_NOT_REACHED
            if target_matches_greedy
            else StrategyComparisonStatus.NOT_COMPARABLE
        )
        failure_reason = (
            None
            if rl_preserved and target_matches_greedy
            else "RL_EXACT_TARGET_NOT_REACHED"
            if not rl_preserved
            else "GREEDY_EXACT_TARGET_NOT_PRESERVED"
        )
        artifacts = (
            (final_test_path, report_path)
            if best is not None
            else (report_path,)
        )
        result = self._result(
            source=source,
            function_name=function_name,
            status=status,
            comparable=comparable,
            reason=failure_reason if not comparable else None,
            scenarios=scenarios,
            full_pool=full_pool_coverage,
            greedy=greedy_result,
            session=session_result,
            hyperparameters=rl_hyperparameters,
            random_seed=random_seed,
            traces=traces,
            exact_count=len(exact_traces),
            best=best,
            rl_verified=(coverage_by_episode[best.episode_number] if best else None),
            rl_duration=self._trace_duration(traces),
            winner=winner,
            equality=target_matches_greedy and rl_preserved,
            artifacts=artifacts,
            failure_category=failure_reason,
            failure_message=(
                None
                if failure_reason is None
                else "Exact coverage eşitliği iki stratejide birlikte doğrulanamadı."
            ),
        )
        return self._write_report(result, report_path)

    @staticmethod
    def _build_traces(
        session: TrainingSessionResult,
        scenarios: tuple[Scenario, ...],
    ) -> tuple[EpisodeSelectionTrace, ...]:
        traces: list[EpisodeSelectionTrace] = []
        for episode in session.episodes:
            identifiers: list[str] = []
            for index in episode.ordered_action_indices:
                if index >= len(scenarios):
                    raise RuntimeError("Episode action scenario havuzu dışında.")
                identifiers.append(scenarios[index].scenario_id)
            ordered = tuple(identifiers)
            traces.append(
                EpisodeSelectionTrace(
                    episode_number=episode.episode_number,
                    ordered_scenario_ids=ordered,
                    executed_action_count=len(ordered),
                    unique_scenario_count=len(set(ordered)),
                    duplicate_selection_count=len(ordered) - len(set(ordered)),
                    final_reported_coverage=episode.final_coverage_percentage,
                    total_reward=episode.total_reward,
                    done_reason=episode.done_reason,
                    duration_seconds=episode.duration_seconds,
                )
            )
        return tuple(traces)

    @staticmethod
    def _select_scenarios(
        scenarios: tuple[Scenario, ...],
        identifiers: tuple[str, ...],
    ) -> tuple[Scenario, ...]:
        by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        try:
            return tuple(by_id[identifier] for identifier in identifiers)
        except KeyError as error:
            raise RuntimeError("Episode trace bilinmeyen scenario ID içeriyor.") from error

    def _measure(
        self,
        *,
        source: Path,
        module_path: str,
        function_name: str,
        start_line: int,
        end_line: int,
        scenarios: tuple[Scenario, ...],
        output_directory: Path,
        test_file_name: str,
        timeout: float,
    ) -> FunctionCoverageResult:
        result = self._suite_service.measure_scenarios(
            source_file=source,
            module_path=module_path,
            function_name=function_name,
            scenarios=scenarios,
            output_directory=output_directory,
            function_start_line=start_line,
            function_end_line=end_line,
            overwrite=True,
            timeout_seconds=timeout,
            test_file_name=test_file_name,
        )
        if not isinstance(result.coverage, FunctionCoverageResult):
            raise RuntimeError("FunctionCoverageResult gerekli.")
        if not result.coverage.has_branch_details:
            raise RuntimeError("Exact branch identity bulunamadı.")
        return result.coverage

    @staticmethod
    def _preserves_exact(
        coverage: FunctionCoverageResult,
        target_lines: tuple[int, ...],
        target_branches: tuple[tuple[int, int], ...],
    ) -> bool:
        return (
            coverage.success
            and tuple(coverage.covered_lines) == target_lines
            and tuple(coverage.covered_branches or ()) == target_branches
        )

    def _effective_timeout(
        self,
        started: float,
        subprocess_timeout: float,
        total_timeout: float | None,
    ) -> float:
        if total_timeout is None:
            return float(subprocess_timeout)
        remaining = float(total_timeout) - (self._clock() - started)
        if remaining <= 0.0:
            raise CoverageExecutionTimeoutError("Comparison toplam süre sınırı aşıldı.")
        return min(float(subprocess_timeout), remaining)

    @staticmethod
    def _trace_duration(traces: tuple[EpisodeSelectionTrace, ...]) -> float | None:
        values = tuple(trace.duration_seconds for trace in traces)
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values if value is not None)

    @staticmethod
    def _validate_inputs(
        *,
        function_name: str,
        scenarios: tuple[Scenario, ...],
        full_pool_coverage: FunctionCoverageResult,
        timeout_seconds: float,
        comparison_timeout_seconds: float | None,
    ) -> None:
        if not function_name.isidentifier():
            raise ValueError("function_name geçerli identifier olmalıdır.")
        if not scenarios or any(not isinstance(value, Scenario) for value in scenarios):
            raise ValueError("scenarios dolu Scenario tuple'ı olmalıdır.")
        if not isinstance(full_pool_coverage, FunctionCoverageResult):
            raise TypeError("full_pool_coverage FunctionCoverageResult olmalıdır.")
        if not full_pool_coverage.success or not full_pool_coverage.has_branch_details:
            raise ValueError("Full-pool exact coverage başarılı olmalıdır.")
        for value, name in (
            (timeout_seconds, "timeout_seconds"),
            (comparison_timeout_seconds, "comparison_timeout_seconds"),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} sayısal veya None olmalıdır.")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} pozitif ve sonlu olmalıdır.")

    @staticmethod
    def _result(
        *,
        source: Path,
        function_name: str,
        status: StrategyComparisonStatus,
        comparable: bool,
        reason: str | None,
        scenarios: tuple[Scenario, ...],
        full_pool: FunctionCoverageResult,
        greedy: ScenarioMinimizationResult,
        session: TrainingSessionResult,
        hyperparameters: tuple[tuple[str, object], ...],
        random_seed: int | None,
        traces: tuple[EpisodeSelectionTrace, ...],
        winner: StrategyWinner,
        equality: bool,
        artifacts: tuple[Path, ...],
        exact_count: int = 0,
        best: EpisodeSelectionTrace | None = None,
        rl_verified: FunctionCoverageResult | None = None,
        rl_duration: float | None = None,
        failure_category: str | None = None,
        failure_message: str | None = None,
    ) -> StrategyComparisonResult:
        return StrategyComparisonResult(
            source_file=source,
            function_name=function_name,
            status=status,
            comparable=comparable,
            non_comparable_reason=reason,
            scenario_pool_count=len(scenarios),
            target_line_identities=tuple(full_pool.covered_lines),
            target_line_percentage=full_pool.line_coverage_percent,
            target_branch_identities=tuple(full_pool.covered_branches or ()),
            target_branch_percentage=full_pool.branch_coverage_percent,
            greedy_algorithm=greedy.algorithm,
            greedy_selected_scenario_ids=greedy.final_selected_scenario_ids,
            greedy_coverage_preserved=greedy.coverage_preserved,
            greedy_duration_seconds=greedy.duration_seconds,
            requested_rl_episode_count=session.requested_episode_count,
            completed_rl_episode_count=session.completed_episode_count,
            rl_hyperparameters=hyperparameters,
            random_seed=random_seed,
            episode_traces=traces,
            exact_target_reaching_episode_count=exact_count,
            best_rl_episode_number=best.episode_number if best else None,
            best_rl_ordered_scenario_ids=best.ordered_scenario_ids if best else (),
            best_rl_executed_test_count=best.executed_action_count if best else None,
            best_rl_unique_scenario_count=best.unique_scenario_count if best else None,
            best_rl_duplicate_count=best.duplicate_selection_count if best else None,
            best_rl_total_reward=best.total_reward if best else None,
            rl_coverage_preserved=best is not None,
            rl_duration_seconds=rl_duration,
            winner=winner,
            coverage_equality_verified=equality,
            artifact_paths=artifacts,
            failure_category=failure_category,
            failure_message=failure_message,
            greedy_verified_line_identities=greedy.final_verified_line_identities,
            greedy_verified_branch_identities=greedy.final_verified_branch_identities,
            rl_verified_line_identities=(
                tuple(rl_verified.covered_lines) if rl_verified else ()
            ),
            rl_verified_branch_identities=(
                tuple(rl_verified.covered_branches or ()) if rl_verified else ()
            ),
        )

    @staticmethod
    def _write_report(
        result: StrategyComparisonResult,
        report_path: Path,
    ) -> StrategyComparisonResult:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, report_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return result
