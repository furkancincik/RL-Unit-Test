from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from generator.file_writer import GeneratedTestFileWriter
from generator.project_pytest_generator import ProjectPytestGenerator
from models.project_coverage_result import (
    ProjectBranchIdentity,
    ProjectCoverageResult,
    ProjectCoverageScopeSummary,
    ProjectCoverageStatus,
    ProjectLineIdentity,
    ProjectTestCandidate,
)
from services.coverage_service import CoverageExecutionTimeoutError, CoverageService


LineTuple = tuple[str, int]
BranchTuple = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class ProjectScenarioCoverageSignature:
    project_test_id: str
    original_order: int
    covered_line_identities: tuple[LineTuple, ...]
    covered_branch_identities: tuple[BranchTuple, ...]
    execution_success: bool

    def __post_init__(self) -> None:
        if not self.project_test_id:
            raise ValueError("project_test_id boş olamaz.")
        if self.original_order < 1:
            raise ValueError("original_order pozitif olmalıdır.")
        if tuple(sorted(set(self.covered_line_identities))) != self.covered_line_identities:
            raise ValueError("Line identity değerleri sıralı ve benzersiz olmalıdır.")
        if tuple(sorted(set(self.covered_branch_identities))) != self.covered_branch_identities:
            raise ValueError("Branch identity değerleri sıralı ve benzersiz olmalıdır.")


@dataclass(frozen=True, slots=True)
class ProjectExactCoverageSelection:
    selected_project_test_ids: tuple[str, ...]
    initial_selected_count: int
    redundancy_removed_count: int


class ProjectExactCoverageGreedySelector:
    """Module-qualified exact identity evreninde deterministik greedy seçer."""

    @classmethod
    def select(
        cls,
        *,
        signatures: tuple[ProjectScenarioCoverageSignature, ...],
        target_lines: tuple[LineTuple, ...],
        target_branches: tuple[BranchTuple, ...],
    ) -> ProjectExactCoverageSelection:
        remaining = [item for item in signatures if item.execution_success]
        selected: list[ProjectScenarioCoverageSignature] = []
        uncovered_lines = set(target_lines)
        uncovered_branches = set(target_branches)
        while uncovered_lines or uncovered_branches:
            ranked: list[
                tuple[
                    tuple[int, int, int, int, str],
                    ProjectScenarioCoverageSignature,
                ]
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
                                item.original_order,
                                item.project_test_id,
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
        kept = list(selected)
        removed_count = 0
        for candidate in reversed(selected):
            others = [item for item in kept if item is not candidate]
            line_union = {
                value for item in others for value in item.covered_line_identities
            }
            branch_union = {
                value for item in others for value in item.covered_branch_identities
            }
            if set(target_lines) <= line_union and set(target_branches) <= branch_union:
                kept.remove(candidate)
                removed_count += 1

        ordered = tuple(
            item.project_test_id
            for item in sorted(
                kept,
                key=lambda value: (value.original_order, value.project_test_id),
            )
        )
        return ProjectExactCoverageSelection(
            selected_project_test_ids=ordered,
            initial_selected_count=initial_count,
            redundancy_removed_count=removed_count,
        )


@dataclass(frozen=True, slots=True)
class _ProjectMeasurement:
    test_exit_code: int
    duration_seconds: float
    executable_lines: tuple[LineTuple, ...]
    executable_branches: tuple[BranchTuple, ...]
    covered_lines: tuple[LineTuple, ...]
    covered_branches: tuple[BranchTuple, ...]


class ProjectCoverageService:
    """Concrete-valid project adaylarını gerçek combined pytest ile ölçer."""

    def __init__(
        self,
        *,
        generator: ProjectPytestGenerator | None = None,
        writer: GeneratedTestFileWriter | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._generator = generator or ProjectPytestGenerator()
        self._writer = writer or GeneratedTestFileWriter()
        self._clock = clock

    def measure_and_minimize(
        self,
        *,
        candidates: tuple[ProjectTestCandidate, ...],
        scope: ProjectCoverageScopeSummary,
        output_root: str | Path,
        timeout_seconds: float = 30.0,
    ) -> ProjectCoverageResult:
        started = self._clock()
        values = self._validate_candidates(candidates)
        if not isinstance(scope, ProjectCoverageScopeSummary):
            raise TypeError("scope ProjectCoverageScopeSummary olmalıdır.")
        timeout = self._validate_timeout(timeout_seconds)
        root = Path(output_root).resolve()
        work = (root / "project_combined").resolve()
        if not work.is_relative_to(root):
            raise RuntimeError("Project combined output root dışına çıktı.")
        full_test = work / "test_project_full_pool.py"
        minimized_test = work / "test_project_greedy_minimized.py"
        report = work / "project_coverage_report.json"

        if not values:
            result = self._build_result(
                status=ProjectCoverageStatus.FAILED,
                scope=scope,
                candidates=values,
                started=started,
                root=root,
                full_test=full_test,
                minimized_test=minimized_test,
                report=report,
                failure_category="EMPTY_PROJECT_SCENARIO_POOL",
                failure_message="Concrete-valid project scenario havuzu boş.",
                unmeasured_reason="NO_USABLE_PROJECT_TESTS",
            )
            return self._write_report(result)

        self._write_suite(values, full_test)
        try:
            full = self._measure(
                values,
                full_test,
                self._remaining_timeout(started, timeout),
            )
        except CoverageExecutionTimeoutError as error:
            result = self._build_result(
                status=ProjectCoverageStatus.TIMED_OUT,
                scope=scope,
                candidates=values,
                started=started,
                root=root,
                full_test=full_test,
                minimized_test=minimized_test,
                report=report,
                failure_category="FULL_PROJECT_SUITE_TIMEOUT",
                failure_message=str(error),
            )
            return self._write_report(result)
        if full.test_exit_code != 0:
            result = self._build_result(
                status=ProjectCoverageStatus.FAILED,
                scope=scope,
                candidates=values,
                started=started,
                root=root,
                full_test=full_test,
                minimized_test=minimized_test,
                report=report,
                full=full,
                failure_category="FULL_PROJECT_SUITE_FAILED",
                failure_message=f"Full combined pytest exit code: {full.test_exit_code}",
            )
            return self._write_report(result)

        target_lines = full.covered_lines
        target_branches = full.covered_branches
        signatures: list[ProjectScenarioCoverageSignature] = []
        cache: dict[tuple[object, ...], _ProjectMeasurement] = {}
        for index, candidate in enumerate(values, start=1):
            precomputed = self._precomputed_signature(candidate)
            if precomputed is not None:
                signatures.append(precomputed)
                continue
            key = self._cache_key(candidate, timeout)
            measurement = cache.get(key)
            if measurement is None:
                contribution_file = (
                    work
                    / "measurements"
                    / f"{index:04d}"
                    / "test_project_contribution.py"
                )
                self._write_suite((candidate,), contribution_file)
                try:
                    measurement = self._measure(
                        (candidate,),
                        contribution_file,
                        self._remaining_timeout(started, timeout),
                    )
                except CoverageExecutionTimeoutError as error:
                    result = self._build_result(
                        status=ProjectCoverageStatus.TIMED_OUT,
                        scope=scope,
                        candidates=values,
                        started=started,
                        root=root,
                        full_test=full_test,
                        minimized_test=minimized_test,
                        report=report,
                        full=full,
                        failure_category="PROJECT_CONTRIBUTION_TIMEOUT",
                        failure_message=str(error),
                    )
                    return self._write_report(result)
                cache[key] = measurement
            signatures.append(
                ProjectScenarioCoverageSignature(
                    project_test_id=candidate.project_test_id,
                    original_order=candidate.original_order,
                    covered_line_identities=tuple(
                        sorted(set(measurement.covered_lines) & set(target_lines))
                    ),
                    covered_branch_identities=tuple(
                        sorted(set(measurement.covered_branches) & set(target_branches))
                    ),
                    execution_success=measurement.test_exit_code == 0,
                )
            )

        selection = ProjectExactCoverageGreedySelector.select(
            signatures=tuple(signatures),
            target_lines=target_lines,
            target_branches=target_branches,
        )
        selected_ids = selection.selected_project_test_ids
        signature_lines = {
            value
            for item in signatures
            if item.project_test_id in set(selected_ids)
            for value in item.covered_line_identities
        }
        signature_branches = {
            value
            for item in signatures
            if item.project_test_id in set(selected_ids)
            for value in item.covered_branch_identities
        }
        additive = (
            set(target_lines) <= signature_lines
            and set(target_branches) <= signature_branches
        )
        if additive:
            selected = tuple(
                item for item in values if item.project_test_id in set(selected_ids)
            )
        else:
            # Interaction-dependent coverage için doğrulanabilir en güvenli suite,
            # gerçek full pool'dur; greedy optimum iddiası üretilmez.
            selected = values
            selected_ids = tuple(item.project_test_id for item in values)

        self._write_suite(selected, minimized_test)
        try:
            minimized = self._measure(
                selected,
                minimized_test,
                self._remaining_timeout(started, timeout),
            )
        except CoverageExecutionTimeoutError as error:
            result = self._build_result(
                status=ProjectCoverageStatus.TIMED_OUT,
                scope=scope,
                candidates=values,
                started=started,
                root=root,
                full_test=full_test,
                minimized_test=minimized_test,
                report=report,
                full=full,
                selected_ids=selected_ids,
                greedy_initial=selection.initial_selected_count,
                redundancy_removed=selection.redundancy_removed_count,
                failure_category="MINIMIZED_PROJECT_SUITE_TIMEOUT",
                failure_message=str(error),
            )
            return self._write_report(result)

        preserved = (
            minimized.test_exit_code == 0
            and minimized.covered_lines == target_lines
            and minimized.covered_branches == target_branches
        )
        if not additive:
            status = ProjectCoverageStatus.NON_ADDITIVE_COVERAGE
            failure_category = "NON_ADDITIVE_COVERAGE"
            failure_message = (
                "Isolated project scenario contributions full combined targetı "
                "açıklamıyor; full pool güvenli fallback olarak doğrulandı."
            )
        elif not preserved:
            status = ProjectCoverageStatus.NON_ADDITIVE_COVERAGE
            failure_category = "FINAL_EXACT_COVERAGE_MISMATCH"
            failure_message = "Minimized suite exact full-pool targetını korumadı."
        else:
            status = (
                ProjectCoverageStatus.COMPLETED
                if scope.scope_complete
                else ProjectCoverageStatus.PARTIAL
            )
            failure_category = None
            failure_message = None
        removed_ids = tuple(
            item.project_test_id
            for item in values
            if item.project_test_id not in set(selected_ids)
        )
        result = self._build_result(
            status=status,
            scope=scope,
            candidates=values,
            started=started,
            root=root,
            full_test=full_test,
            minimized_test=minimized_test,
            report=report,
            full=full,
            minimized=minimized,
            selected_ids=selected_ids,
            removed_ids=removed_ids,
            greedy_initial=selection.initial_selected_count,
            redundancy_removed=selection.redundancy_removed_count,
            coverage_preserved=preserved,
            failure_category=failure_category,
            failure_message=failure_message,
        )
        return self._write_report(result)

    def _write_suite(
        self,
        candidates: tuple[ProjectTestCandidate, ...],
        path: Path,
    ) -> None:
        self._writer.write(
            self._generator.generate(candidates),
            path,
            overwrite=True,
        )

    def _measure(
        self,
        candidates: tuple[ProjectTestCandidate, ...],
        test_file: Path,
        timeout: float,
    ) -> _ProjectMeasurement:
        started = self._clock()
        sources = tuple(dict.fromkeys(item.source_file.resolve() for item in candidates))
        roots = tuple(dict.fromkeys(item.import_root.resolve() for item in candidates))
        include = ",".join(path.as_posix() for path in sources)
        working_directory = roots[0]
        with tempfile.TemporaryDirectory(prefix="rl_unit_test_project_coverage_") as raw:
            temporary = Path(raw)
            coverage_data = temporary / ".coverage"
            coverage_json = temporary / "coverage.json"
            environment = os.environ.copy()
            python_path_parts = [str(path) for path in roots]
            current_python_path = environment.get("PYTHONPATH")
            if current_python_path:
                python_path_parts.append(current_python_path)
            environment["PYTHONPATH"] = os.pathsep.join(python_path_parts)
            environment["COVERAGE_FILE"] = str(coverage_data)
            run_command = (
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--branch",
                f"--include={include}",
                "-m",
                "pytest",
                str(test_file.resolve()),
                "-q",
                "--color=no",
            )
            try:
                run = subprocess.run(
                    run_command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                    cwd=working_directory,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise CoverageExecutionTimeoutError(
                    "Combined project pytest/coverage zaman aşımına uğradı."
                ) from error
            remaining = timeout - (self._clock() - started)
            if remaining <= 0:
                raise CoverageExecutionTimeoutError(
                    "Combined project coverage raporu zaman aşımına uğradı."
                )
            report_command = (
                sys.executable,
                "-m",
                "coverage",
                "json",
                "-o",
                str(coverage_json),
                f"--include={include}",
            )
            try:
                report = subprocess.run(
                    report_command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=remaining,
                    check=False,
                    cwd=working_directory,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise CoverageExecutionTimeoutError(
                    "Combined project coverage raporu zaman aşımına uğradı."
                ) from error
            if report.returncode != 0:
                raise RuntimeError(
                    "Combined project coverage JSON raporu oluşturulamadı: "
                    f"{report.stderr.strip()}"
                )
            report_data = CoverageService._read_report(coverage_json)
            executable_lines: set[LineTuple] = set()
            executable_branches: set[BranchTuple] = set()
            covered_lines: set[LineTuple] = set()
            covered_branches: set[BranchTuple] = set()
            candidates_by_source: dict[Path, list[ProjectTestCandidate]] = {}
            for candidate in candidates:
                candidates_by_source.setdefault(candidate.source_file.resolve(), []).append(candidate)
            for source, source_candidates in candidates_by_source.items():
                file_data = CoverageService._find_file_data(
                    report_data,
                    source,
                    working_directory,
                )
                executed = CoverageService._read_line_collection(
                    file_data, "executed_lines"
                )
                missing = CoverageService._read_line_collection(
                    file_data, "missing_lines"
                )
                executed_branches = CoverageService._read_branch_collection(
                    file_data, "executed_branches"
                )
                missing_branches = CoverageService._read_branch_collection(
                    file_data, "missing_branches"
                )
                for candidate in source_candidates:
                    relative = candidate.relative_module_path
                    start = candidate.function_start_line
                    end = candidate.function_end_line
                    covered_lines.update(
                        (relative, line) for line in executed if start <= line <= end
                    )
                    executable_lines.update(
                        (relative, line)
                        for line in (*executed, *missing)
                        if start <= line <= end
                    )
                    covered_branches.update(
                        (relative, branch[0], branch[1])
                        for branch in executed_branches
                        if start <= branch[0] <= end
                    )
                    executable_branches.update(
                        (relative, branch[0], branch[1])
                        for branch in (*executed_branches, *missing_branches)
                        if start <= branch[0] <= end
                    )
        return _ProjectMeasurement(
            test_exit_code=run.returncode,
            duration_seconds=max(0.0, self._clock() - started),
            executable_lines=tuple(sorted(executable_lines)),
            executable_branches=tuple(sorted(executable_branches)),
            covered_lines=tuple(sorted(covered_lines)),
            covered_branches=tuple(sorted(covered_branches)),
        )

    @staticmethod
    def _precomputed_signature(
        candidate: ProjectTestCandidate,
    ) -> ProjectScenarioCoverageSignature | None:
        if candidate.precomputed_execution_success is None:
            return None
        relative = candidate.relative_module_path
        return ProjectScenarioCoverageSignature(
            project_test_id=candidate.project_test_id,
            original_order=candidate.original_order,
            covered_line_identities=tuple(
                sorted(
                    (relative, line)
                    for line in (candidate.precomputed_line_identities or ())
                )
            ),
            covered_branch_identities=tuple(
                sorted(
                    (relative, branch[0], branch[1])
                    for branch in (candidate.precomputed_branch_identities or ())
                )
            ),
            execution_success=candidate.precomputed_execution_success,
        )

    @staticmethod
    def _cache_key(
        candidate: ProjectTestCandidate,
        timeout: float,
    ) -> tuple[object, ...]:
        stat = candidate.source_file.stat()
        return (
            str(candidate.source_file.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            candidate.module_path,
            candidate.function_name,
            candidate.function_start_line,
            candidate.function_end_line,
            candidate.scenario.scenario_id,
            float(timeout),
        )

    @staticmethod
    def _validate_candidates(
        candidates: tuple[ProjectTestCandidate, ...],
    ) -> tuple[ProjectTestCandidate, ...]:
        if not isinstance(candidates, tuple):
            raise TypeError("candidates tuple olmalıdır.")
        if any(not isinstance(value, ProjectTestCandidate) for value in candidates):
            raise TypeError("candidates yalnız ProjectTestCandidate içermelidir.")
        identifiers = tuple(value.project_test_id for value in candidates)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("project_test_id değerleri benzersiz olmalıdır.")
        orders = tuple(value.original_order for value in candidates)
        if len(set(orders)) != len(orders):
            raise ValueError("original_order değerleri benzersiz olmalıdır.")
        return tuple(sorted(candidates, key=lambda value: (value.original_order, value.project_test_id)))

    @staticmethod
    def _validate_timeout(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("timeout_seconds pozitif sonlu sayı olmalıdır.")
        return float(value)

    def _remaining_timeout(self, started: float, timeout: float) -> float:
        remaining = timeout - (self._clock() - started)
        if remaining <= 0:
            raise CoverageExecutionTimeoutError(
                "Combined project coverage toplam süre sınırı aşıldı."
            )
        return remaining

    @staticmethod
    def _percentage(covered: int, executable: int) -> float:
        if executable == 0:
            return 100.0
        return covered / executable * 100.0

    def _build_result(
        self,
        *,
        status: ProjectCoverageStatus,
        scope: ProjectCoverageScopeSummary,
        candidates: tuple[ProjectTestCandidate, ...],
        started: float,
        root: Path,
        full_test: Path,
        minimized_test: Path,
        report: Path,
        full: _ProjectMeasurement | None = None,
        minimized: _ProjectMeasurement | None = None,
        selected_ids: tuple[str, ...] = (),
        removed_ids: tuple[str, ...] = (),
        greedy_initial: int = 0,
        redundancy_removed: int = 0,
        coverage_preserved: bool = False,
        failure_category: str | None = None,
        failure_message: str | None = None,
        unmeasured_reason: str | None = None,
    ) -> ProjectCoverageResult:
        full_lines = full.covered_lines if full else ()
        full_branches = full.covered_branches if full else ()
        executable_lines = full.executable_lines if full else ()
        executable_branches = full.executable_branches if full else ()
        minimized_lines = minimized.covered_lines if minimized else ()
        minimized_branches = minimized.covered_branches if minimized else ()
        artifacts = tuple(
            path for path in (full_test, minimized_test, report) if path.exists() or path == report
        )
        return ProjectCoverageResult(
            status=status,
            scope=scope,
            executable_line_identities=tuple(
                ProjectLineIdentity(*value) for value in executable_lines
            ),
            executable_branch_identities=tuple(
                ProjectBranchIdentity(*value) for value in executable_branches
            ),
            target_line_identities=tuple(
                ProjectLineIdentity(*value) for value in full_lines
            ),
            target_branch_identities=tuple(
                ProjectBranchIdentity(*value) for value in full_branches
            ),
            minimized_covered_line_identities=tuple(
                ProjectLineIdentity(*value) for value in minimized_lines
            ),
            minimized_covered_branch_identities=tuple(
                ProjectBranchIdentity(*value) for value in minimized_branches
            ),
            full_line_coverage_percent=(
                self._percentage(len(full_lines), len(executable_lines)) if full else None
            ),
            full_branch_coverage_percent=(
                self._percentage(len(full_branches), len(executable_branches)) if full else None
            ),
            minimized_line_coverage_percent=(
                self._percentage(len(minimized_lines), len(executable_lines))
                if minimized
                else None
            ),
            minimized_branch_coverage_percent=(
                self._percentage(len(minimized_branches), len(executable_branches))
                if minimized
                else None
            ),
            full_pytest_exit_code=full.test_exit_code if full else None,
            minimized_pytest_exit_code=minimized.test_exit_code if minimized else None,
            full_scenario_count=len(candidates),
            greedy_initially_selected_count=greedy_initial,
            redundancy_elimination_removed_count=redundancy_removed,
            selected_project_test_ids=selected_ids,
            removed_project_test_ids=removed_ids,
            coverage_preserved=coverage_preserved,
            duration_seconds=max(0.0, self._clock() - started),
            output_root=root,
            full_test_file=full_test,
            minimized_test_file=minimized_test,
            report_path=report,
            artifact_paths=artifacts,
            failure_category=failure_category,
            failure_message=failure_message,
            unmeasured_reason=unmeasured_reason,
        )

    @staticmethod
    def _write_report(result: ProjectCoverageResult) -> ProjectCoverageResult:
        result.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result.report_path.with_name(
            f".{result.report_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, result.report_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return result
