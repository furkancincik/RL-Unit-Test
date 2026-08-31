from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from generator.scenario_generator import Scenario
from models.coverage_result import FunctionCoverageResult
from models.scenario_minimization_result import (
    CoverageMeasurementStatus,
    ScenarioCoverageSignature,
    ScenarioMinimizationResult,
    ScenarioMinimizationStatus,
)
from services.coverage_service import CoverageExecutionTimeoutError
from services.scenario_suite_coverage_service import ScenarioSuiteCoverageService


@dataclass(frozen=True, slots=True)
class ExactCoverageSelection:
    selected_scenario_ids: tuple[str, ...]
    initial_selected_count: int
    redundancy_removed_count: int


class ExactCoverageGreedySelector:
    """Exact line/branch identity evreni için deterministik set-cover."""

    @classmethod
    def select(
        cls,
        *,
        signatures: tuple[ScenarioCoverageSignature, ...],
        target_lines: tuple[int, ...],
        target_branches: tuple[tuple[int, int], ...],
    ) -> ExactCoverageSelection:
        remaining = [item for item in signatures if item.execution_success]
        selected: list[ScenarioCoverageSignature] = []
        uncovered_lines = set(target_lines)
        uncovered_branches = set(target_branches)
        while uncovered_lines or uncovered_branches:
            ranked: list[
                tuple[tuple[int, int, int, int, str], ScenarioCoverageSignature]
            ] = []
            for item in remaining:
                new_lines = uncovered_lines & set(item.covered_line_identities)
                new_branches = uncovered_branches & set(item.covered_branch_identities)
                total = len(new_lines) + len(new_branches)
                if total:
                    ranked.append(
                        (
                            (
                                -total,
                                -len(new_branches),
                                -len(new_lines),
                                item.original_scenario_index,
                                item.scenario_id,
                            ),
                            item,
                        )
                    )
            if not ranked:
                break
            chosen = min(ranked, key=lambda entry: entry[0])[1]
            selected.append(chosen)
            remaining.remove(chosen)
            uncovered_lines.difference_update(chosen.covered_line_identities)
            uncovered_branches.difference_update(chosen.covered_branch_identities)

        initial_count = len(selected)
        selected_tuple, removed = cls.eliminate_redundancy(
            selected=tuple(selected),
            target_lines=target_lines,
            target_branches=target_branches,
        )
        ordered = tuple(
            item.scenario_id
            for item in sorted(
                selected_tuple,
                key=lambda value: (value.original_scenario_index, value.scenario_id),
            )
        )
        return ExactCoverageSelection(
            selected_scenario_ids=ordered,
            initial_selected_count=initial_count,
            redundancy_removed_count=len(removed),
        )

    @staticmethod
    def eliminate_redundancy(
        *,
        selected: tuple[ScenarioCoverageSignature, ...],
        target_lines: tuple[int, ...],
        target_branches: tuple[tuple[int, int], ...],
    ) -> tuple[
        tuple[ScenarioCoverageSignature, ...],
        tuple[ScenarioCoverageSignature, ...],
    ]:
        kept = list(selected)
        removed: list[ScenarioCoverageSignature] = []
        for candidate in reversed(selected):
            others = [item for item in kept if item is not candidate]
            line_union = {line for item in others for line in item.covered_line_identities}
            branch_union = {
                branch for item in others for branch in item.covered_branch_identities
            }
            if set(target_lines) <= line_union and set(target_branches) <= branch_union:
                kept.remove(candidate)
                removed.append(candidate)
        return tuple(kept), tuple(removed)


class ScenarioCoverageMinimizationService:
    """Validated scenario pool için gerçek exact-coverage greedy baseline."""

    def __init__(
        self,
        *,
        suite_service: ScenarioSuiteCoverageService | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._suite_service = suite_service or ScenarioSuiteCoverageService()
        self._clock = clock

    def minimize(
        self,
        *,
        source_file: str | Path,
        module_path: str,
        function_name: str,
        function_start_line: int,
        function_end_line: int,
        scenarios: tuple[Scenario, ...],
        output_root: str | Path,
        timeout_seconds: float = 30.0,
        minimization_timeout_seconds: float | None = None,
        full_pool_coverage: FunctionCoverageResult | None = None,
        import_root: str | Path | None = None,
    ) -> ScenarioMinimizationResult:
        started = self._clock()
        source = Path(source_file).resolve()
        root = Path(output_root).resolve()
        work_directory = (root / "greedy_minimized").resolve()
        if not work_directory.is_relative_to(root):
            raise RuntimeError("Greedy output yolu root dışına çıktı.")
        if not function_name.isidentifier():
            raise ValueError("function_name geçerli identifier olmalıdır.")
        if not isinstance(scenarios, tuple):
            raise TypeError("scenarios tuple olmalıdır.")
        if any(not isinstance(scenario, Scenario) for scenario in scenarios):
            raise TypeError("scenarios yalnız Scenario nesneleri içermelidir.")
        scenario_ids = tuple(scenario.scenario_id for scenario in scenarios)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenarios tekrar eden scenario_id içeremez.")
        self._validate_timeout(timeout_seconds, "timeout_seconds")
        if minimization_timeout_seconds is not None:
            self._validate_timeout(
                minimization_timeout_seconds, "minimization_timeout_seconds"
            )
        report_path = work_directory / "minimization_report.json"
        if not scenarios:
            result = self._result(
                source=source,
                function_name=function_name,
                status=ScenarioMinimizationStatus.EMPTY_POOL,
                scenarios=scenarios,
                contributions=(),
                duration=self._clock() - started,
                report_path=report_path,
                failure_category="EMPTY_SCENARIO_POOL",
                failure_message="Concrete-valid scenario havuzu boş.",
            )
            return self._write_report(result, report_path)

        full_pool_result = full_pool_coverage
        try:
            if full_pool_result is None:
                full_pool_result = self._measure(
                    source=source,
                    module_path=module_path,
                    function_name=function_name,
                    start_line=function_start_line,
                    end_line=function_end_line,
                    scenarios=scenarios,
                    output_directory=work_directory / "full_pool",
                    test_file_name=f"test_{function_name}_full_pool.py",
                    timeout=self._effective_timeout(
                        started, timeout_seconds, minimization_timeout_seconds
                    ),
                    import_root=import_root,
                )
        except CoverageExecutionTimeoutError as error:
            result = self._result(
                source=source,
                function_name=function_name,
                status=ScenarioMinimizationStatus.TIMED_OUT,
                scenarios=scenarios,
                contributions=(),
                duration=self._clock() - started,
                report_path=report_path,
                failure_category="FULL_POOL_TIMEOUT",
                failure_message=str(error),
            )
            return self._write_report(result, report_path)

        if not full_pool_result.success:
            result = self._result(
                source=source,
                function_name=function_name,
                status=ScenarioMinimizationStatus.FULL_POOL_FAILED,
                scenarios=scenarios,
                contributions=(),
                duration=self._clock() - started,
                report_path=report_path,
                target=full_pool_result,
                failure_category="FULL_POOL_EXECUTION_FAILED",
                failure_message=f"Full pool pytest exit code: {full_pool_result.test_exit_code}",
            )
            return self._write_report(result, report_path)
        if not full_pool_result.has_branch_details:
            raise RuntimeError("Exact branch identity içeren function coverage gereklidir.")

        contributions: list[ScenarioCoverageSignature] = []
        cache: dict[tuple[object, ...], FunctionCoverageResult] = {}
        for index, scenario in enumerate(scenarios, start=1):
            try:
                remaining = self._effective_timeout(
                    started, timeout_seconds, minimization_timeout_seconds
                )
                key = self._cache_key(
                    source=source,
                    function_name=function_name,
                    start_line=function_start_line,
                    end_line=function_end_line,
                    timeout_seconds=timeout_seconds,
                    scenario=scenario,
                )
                coverage = cache.get(key)
                if coverage is None:
                    coverage = self._measure(
                        source=source,
                        module_path=module_path,
                        function_name=function_name,
                        start_line=function_start_line,
                        end_line=function_end_line,
                        scenarios=(scenario,),
                        output_directory=work_directory / "measurements" / f"{index:04d}",
                        test_file_name=f"test_{function_name}_contribution.py",
                        timeout=remaining,
                        import_root=import_root,
                    )
                    cache[key] = coverage
                contributions.append(self._signature(scenario, index, coverage))
                partial = self._result(
                    source=source,
                    function_name=function_name,
                    status=ScenarioMinimizationStatus.PARTIAL,
                    scenarios=scenarios,
                    contributions=tuple(contributions),
                    duration=self._clock() - started,
                    report_path=report_path,
                    target=full_pool_result,
                    failure_category="MEASUREMENT_IN_PROGRESS",
                    failure_message="Scenario contribution ölçümü devam ediyor.",
                )
                self._write_report(partial, report_path)
            except CoverageExecutionTimeoutError as error:
                contributions.append(
                    ScenarioCoverageSignature(
                        scenario_id=scenario.scenario_id,
                        original_scenario_index=index,
                        function_name=function_name,
                        covered_line_identities=(),
                        covered_branch_identities=(),
                        line_contribution_count=0,
                        branch_contribution_count=0,
                        execution_success=False,
                        test_exit_code=None,
                        duration_seconds=None,
                        measurement_status=CoverageMeasurementStatus.TIMED_OUT,
                        failure_category="SCENARIO_TIMEOUT",
                        failure_message=str(error),
                    )
                )
                result = self._result(
                    source=source,
                    function_name=function_name,
                    status=ScenarioMinimizationStatus.TIMED_OUT,
                    scenarios=scenarios,
                    contributions=tuple(contributions),
                    duration=self._clock() - started,
                    report_path=report_path,
                    target=full_pool_result,
                    failure_category="MINIMIZATION_TIMEOUT",
                    failure_message=str(error),
                )
                return self._write_report(result, report_path)

        target_lines = tuple(full_pool_result.covered_lines)
        target_branches = tuple(full_pool_result.covered_branches or ())
        selection = ExactCoverageGreedySelector.select(
            signatures=tuple(contributions),
            target_lines=target_lines,
            target_branches=target_branches,
        )
        selected_ids = selection.selected_scenario_ids
        selected = tuple(
            scenario for scenario in scenarios if scenario.scenario_id in set(selected_ids)
        )
        covered_lines = {
            line
            for item in contributions
            if item.scenario_id in set(selected_ids)
            for line in item.covered_line_identities
        }
        covered_branches = {
            branch
            for item in contributions
            if item.scenario_id in set(selected_ids)
            for branch in item.covered_branch_identities
        }
        if not set(target_lines) <= covered_lines or not set(target_branches) <= covered_branches:
            result = self._result(
                source=source,
                function_name=function_name,
                status=ScenarioMinimizationStatus.TARGET_NOT_COVERED,
                scenarios=scenarios,
                contributions=tuple(contributions),
                duration=self._clock() - started,
                report_path=report_path,
                target=full_pool_result,
                initial_count=selection.initial_selected_count,
                selected_ids=selected_ids,
                failure_category="ISOLATED_UNION_INCOMPLETE",
                failure_message="Isolated contribution union full-pool targetını kapsamıyor.",
            )
            return self._write_report(result, report_path)

        try:
            final_coverage = self._measure(
                source=source,
                module_path=module_path,
                function_name=function_name,
                start_line=function_start_line,
                end_line=function_end_line,
                scenarios=selected,
                output_directory=work_directory,
                test_file_name=f"test_{function_name}_greedy_minimized.py",
                timeout=self._effective_timeout(
                    started, timeout_seconds, minimization_timeout_seconds
                ),
                import_root=import_root,
            )
        except CoverageExecutionTimeoutError as error:
            result = self._result(
                source=source,
                function_name=function_name,
                status=ScenarioMinimizationStatus.TIMED_OUT,
                scenarios=scenarios,
                contributions=tuple(contributions),
                duration=self._clock() - started,
                report_path=report_path,
                target=full_pool_result,
                initial_count=selection.initial_selected_count,
                selected_ids=selected_ids,
                failure_category="FINAL_VERIFICATION_TIMEOUT",
                failure_message=str(error),
            )
            return self._write_report(result, report_path)

        preserved = (
            final_coverage.success
            and tuple(final_coverage.covered_lines) == target_lines
            and tuple(final_coverage.covered_branches or ()) == target_branches
        )
        all_ids = tuple(scenario.scenario_id for scenario in scenarios)
        removed_ids = tuple(identifier for identifier in all_ids if identifier not in set(selected_ids))
        status = (
            ScenarioMinimizationStatus.COMPLETED
            if preserved and removed_ids
            else ScenarioMinimizationStatus.NO_REDUCTION
            if preserved
            else ScenarioMinimizationStatus.NON_ADDITIVE_COVERAGE
        )
        result = self._result(
            source=source,
            function_name=function_name,
            status=status,
            scenarios=scenarios,
            contributions=tuple(contributions),
            duration=self._clock() - started,
            report_path=report_path,
            target=full_pool_result,
            final=final_coverage,
            initial_count=selection.initial_selected_count,
            redundancy_removed=selection.redundancy_removed_count,
            selected_ids=selected_ids,
            removed_ids=removed_ids,
            coverage_preserved=preserved,
            artifact_paths=(final_coverage.test_file, report_path),
            failure_category=None if preserved else "NON_ADDITIVE_COVERAGE",
            failure_message=None if preserved else "Final suite exact full-pool targetını korumadı.",
        )
        return self._write_report(result, report_path)

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
        import_root: str | Path | None,
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
            import_root=import_root,
        )
        if not isinstance(result.coverage, FunctionCoverageResult):
            raise RuntimeError("FunctionCoverageResult gerekli.")
        if not result.coverage.has_branch_details:
            raise RuntimeError("Exact branch identity bulunamadı.")
        return result.coverage

    @staticmethod
    def _signature(
        scenario: Scenario,
        index: int,
        coverage: FunctionCoverageResult,
    ) -> ScenarioCoverageSignature:
        success = coverage.success
        return ScenarioCoverageSignature(
            scenario_id=scenario.scenario_id,
            original_scenario_index=index,
            function_name=coverage.function_name,
            covered_line_identities=tuple(coverage.covered_lines) if success else (),
            covered_branch_identities=tuple(coverage.covered_branches or ()) if success else (),
            line_contribution_count=len(coverage.covered_lines) if success else 0,
            branch_contribution_count=len(coverage.covered_branches or ()) if success else 0,
            execution_success=success,
            test_exit_code=coverage.test_exit_code,
            duration_seconds=coverage.duration_seconds,
            measurement_status=(
                CoverageMeasurementStatus.SUCCESS
                if success
                else CoverageMeasurementStatus.INTERACTION_DEPENDENT
            ),
            failure_category=None if success else "NON_ISOLATED_SCENARIO",
            failure_message=(
                None
                if success
                else f"Scenario tek başına pytest exit code {coverage.test_exit_code} üretti."
            ),
        )

    @staticmethod
    def _cache_key(
        *,
        source: Path,
        function_name: str,
        start_line: int,
        end_line: int,
        timeout_seconds: float,
        scenario: Scenario,
    ) -> tuple[object, ...]:
        stat = source.stat() if source.exists() else None
        fingerprint = (stat.st_size, stat.st_mtime_ns) if stat is not None else (None, None)
        return (
            str(source),
            fingerprint,
            function_name,
            start_line,
            end_line,
            float(timeout_seconds),
            scenario.execution_identity,
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
            raise CoverageExecutionTimeoutError("Minimization toplam süre sınırı aşıldı.")
        return min(float(subprocess_timeout), remaining)

    @staticmethod
    def _validate_timeout(value: float, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} pozitif sonlu sayı olmalıdır.")

    @staticmethod
    def _result(
        *,
        source: Path,
        function_name: str,
        status: ScenarioMinimizationStatus,
        scenarios: tuple[Scenario, ...],
        contributions: tuple[ScenarioCoverageSignature, ...],
        duration: float,
        report_path: Path,
        target: FunctionCoverageResult | None = None,
        final: FunctionCoverageResult | None = None,
        initial_count: int = 0,
        redundancy_removed: int = 0,
        selected_ids: tuple[str, ...] = (),
        removed_ids: tuple[str, ...] = (),
        coverage_preserved: bool = False,
        artifact_paths: tuple[Path, ...] = (),
        failure_category: str | None = None,
        failure_message: str | None = None,
    ) -> ScenarioMinimizationResult:
        return ScenarioMinimizationResult(
            source_file=source,
            function_name=function_name,
            status=status,
            full_pool_scenario_count=len(scenarios),
            contributions=contributions,
            target_line_identities=tuple(target.covered_lines) if target else (),
            target_branch_identities=tuple(target.covered_branches or ()) if target else (),
            target_line_percentage=target.line_coverage_percent if target else None,
            target_branch_percentage=target.branch_coverage_percent if target else None,
            greedy_initially_selected_count=initial_count,
            redundancy_elimination_removed_count=redundancy_removed,
            final_selected_scenario_ids=selected_ids,
            removed_scenario_ids=removed_ids,
            coverage_preserved=coverage_preserved,
            final_verified_line_identities=tuple(final.covered_lines) if final else (),
            final_verified_branch_identities=tuple(final.covered_branches or ()) if final else (),
            final_verified_line_percentage=final.line_coverage_percent if final else None,
            final_verified_branch_percentage=final.branch_coverage_percent if final else None,
            duration_seconds=max(0.0, duration),
            failure_category=failure_category,
            failure_message=failure_message,
            artifact_paths=artifact_paths or (report_path,),
        )

    @staticmethod
    def _write_report(
        result: ScenarioMinimizationResult,
        report_path: Path,
    ) -> ScenarioMinimizationResult:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(
            f".{report_path.name}.{uuid.uuid4().hex}.tmp"
        )
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
