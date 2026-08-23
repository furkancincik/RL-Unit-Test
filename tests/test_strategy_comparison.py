from __future__ import annotations

import json
import ast
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from models.strategy_comparison_result import (
    EpisodeSelectionTrace,
    StrategyComparisonResult,
    StrategyComparisonStatus,
    StrategyWinner,
)
from generator.scenario_generator import Scenario
from models.coverage_result import CoverageResult, FunctionCoverageResult
from models.scenario_minimization_result import (
    ScenarioMinimizationResult,
    ScenarioMinimizationStatus,
)
from rl.training_session import TrainingSessionResult
from rl.training_statistics import EpisodeStatistics
from services.coverage_service import CoverageExecutionTimeoutError
from services.scenario_suite_coverage_service import ScenarioSuiteCoverageResult
from services.scenario_coverage_minimization_service import (
    ScenarioCoverageMinimizationService,
)
from services.scenario_suite_coverage_service import ScenarioSuiteCoverageService
from services.strategy_comparison_service import (
    ExactRLEpisodeSelector,
    StrategyComparisonService,
)


def _trace(
    episode: int,
    scenario_ids: tuple[str, ...],
    *,
    executed: int | None = None,
    reward: float = 10.0,
    coverage: float = 100.0,
) -> EpisodeSelectionTrace:
    executed_count = len(scenario_ids) if executed is None else executed
    return EpisodeSelectionTrace(
        episode_number=episode,
        ordered_scenario_ids=scenario_ids,
        executed_action_count=executed_count,
        unique_scenario_count=len(set(scenario_ids)),
        duplicate_selection_count=executed_count - len(set(scenario_ids)),
        final_reported_coverage=coverage,
        total_reward=reward,
        done_reason="TARGET_COVERAGE_REACHED",
        duration_seconds=0.1,
    )


def test_episode_trace_is_immutable_and_json_safe() -> None:
    trace = _trace(1, ("first", "second", "first"), executed=3)

    with pytest.raises(FrozenInstanceError):
        trace.episode_number = 2  # type: ignore[misc]

    assert trace.unique_scenario_count == 2
    assert trace.duplicate_selection_count == 1
    assert trace.to_dict()["ordered_scenario_ids"] == ["first", "second", "first"]
    serialized = json.dumps(trace.to_dict())
    assert "keyword_arguments" not in serialized
    assert "expected_result" not in serialized


def test_episode_trace_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="executed_action_count"):
        EpisodeSelectionTrace(
            episode_number=1,
            ordered_scenario_ids=("only",),
            executed_action_count=2,
            unique_scenario_count=1,
            duplicate_selection_count=1,
            final_reported_coverage=100.0,
            total_reward=1.0,
            done_reason="DONE",
            duration_seconds=0.1,
        )


def test_exact_episode_selector_prefers_fewer_executed_actions() -> None:
    selected = ExactRLEpisodeSelector.select_best(
        (
            _trace(1, ("a", "b", "c"), reward=100.0),
            _trace(2, ("a", "b"), reward=1.0),
        )
    )

    assert selected.episode_number == 2


def test_exact_episode_selector_uses_unique_reward_and_episode_ties() -> None:
    fewer_unique = _trace(3, ("a", "a"), executed=2, reward=1.0)
    higher_reward = _trace(2, ("a", "b"), reward=20.0)
    earlier = _trace(1, ("a", "b"), reward=20.0)

    assert ExactRLEpisodeSelector.select_best((higher_reward, fewer_unique)) is fewer_unique
    assert ExactRLEpisodeSelector.select_best((higher_reward, earlier)) is earlier


@pytest.mark.parametrize(
    ("greedy_preserved", "rl_preserved", "greedy_count", "rl_count", "winner"),
    (
        (True, True, 3, 2, StrategyWinner.RL),
        (True, True, 2, 3, StrategyWinner.GREEDY),
        (True, True, 2, 2, StrategyWinner.TIE),
        (False, True, 2, 3, StrategyWinner.RL),
        (True, False, 2, 1, StrategyWinner.GREEDY),
        (False, False, 2, 1, StrategyWinner.NOT_COMPARABLE),
    ),
)
def test_winner_policy(
    greedy_preserved: bool,
    rl_preserved: bool,
    greedy_count: int,
    rl_count: int,
    winner: StrategyWinner,
) -> None:
    assert StrategyComparisonResult.determine_winner(
        greedy_coverage_preserved=greedy_preserved,
        rl_coverage_preserved=rl_preserved,
        greedy_selected_count=greedy_count,
        rl_selected_count=rl_count,
    ) is winner


def test_result_reports_reductions_and_never_claims_global_minimum(
    tmp_path: Path,
) -> None:
    result = StrategyComparisonResult(
        source_file=tmp_path / "target.py",
        function_name="target",
        status=StrategyComparisonStatus.COMPLETED,
        comparable=True,
        non_comparable_reason=None,
        scenario_pool_count=4,
        target_line_identities=(1, 2),
        target_line_percentage=100.0,
        target_branch_identities=((1, 2),),
        target_branch_percentage=100.0,
        greedy_algorithm="GREEDY_SET_COVER_WITH_BACKWARD_ELIMINATION",
        greedy_selected_scenario_ids=("a", "b"),
        greedy_coverage_preserved=True,
        greedy_duration_seconds=1.0,
        requested_rl_episode_count=2,
        completed_rl_episode_count=2,
        rl_hyperparameters=(("epsilon", 0.1),),
        random_seed=42,
        episode_traces=(_trace(1, ("a",)), _trace(2, ("a", "b"))),
        exact_target_reaching_episode_count=1,
        best_rl_episode_number=1,
        best_rl_ordered_scenario_ids=("a",),
        best_rl_executed_test_count=1,
        best_rl_unique_scenario_count=1,
        best_rl_duplicate_count=0,
        best_rl_total_reward=10.0,
        rl_coverage_preserved=True,
        rl_duration_seconds=2.0,
        winner=StrategyWinner.RL,
        coverage_equality_verified=True,
        artifact_paths=(tmp_path / "report.json",),
    )

    assert result.rl_reduction_percentage == 75.0
    assert result.greedy_reduction_percentage == 50.0
    assert result.rl_count_minus_greedy_count == -1
    assert result.globally_minimal is False
    assert result.to_dict()["globally_minimal"] is False


def _scenario(identifier: str, index: int) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        name=identifier,
        path_index=index,
        priority_rank=index,
        priority_level="High",
        dqm_score=100.0,
        node_ids=(1,),
        edge_labels=(),
        contains_loop=False,
        contains_exception=False,
        description="comparison fixture",
        keyword_arguments=(("value", index),),
        expected_result=index,
        expected_exception=None,
    )


def _coverage(
    tmp_path: Path,
    lines: tuple[int, ...],
    branches: tuple[tuple[int, int], ...],
) -> FunctionCoverageResult:
    all_lines = (1, 2, 3)
    all_branches = ((2, 3), (2, -1))
    file_coverage = CoverageResult(
        source_file=tmp_path / "target.py",
        test_file=tmp_path / "test_target.py",
        line_coverage_percent=len(lines) / len(all_lines) * 100.0,
        branch_coverage_percent=len(branches) / len(all_branches) * 100.0,
        covered_line_count=len(lines),
        missing_line_count=len(set(all_lines) - set(lines)),
        total_line_count=len(all_lines),
        covered_branch_count=len(branches),
        missing_branch_count=len(set(all_branches) - set(branches)),
        total_branch_count=len(all_branches),
        test_exit_code=0,
        duration_seconds=0.01,
    )
    return FunctionCoverageResult(
        source_file=file_coverage.source_file,
        test_file=file_coverage.test_file,
        function_name="target",
        start_line=1,
        end_line=3,
        line_coverage_percent=file_coverage.line_coverage_percent,
        branch_coverage_percent=file_coverage.branch_coverage_percent,
        covered_lines=lines,
        missing_lines=tuple(sorted(set(all_lines) - set(lines))),
        covered_branch_count=len(branches),
        missing_branch_count=len(set(all_branches) - set(branches)),
        covered_branches=branches,
        missing_branches=tuple(sorted(set(all_branches) - set(branches))),
        test_exit_code=0,
        duration_seconds=0.01,
        file_coverage=file_coverage,
    )


def _greedy(
    tmp_path: Path,
    full: FunctionCoverageResult,
    selected: tuple[str, ...],
    *,
    preserved: bool = True,
) -> ScenarioMinimizationResult:
    return ScenarioMinimizationResult(
        source_file=tmp_path / "target.py",
        function_name="target",
        status=ScenarioMinimizationStatus.COMPLETED,
        full_pool_scenario_count=3,
        contributions=(),
        target_line_identities=full.covered_lines,
        target_branch_identities=full.covered_branches or (),
        target_line_percentage=full.line_coverage_percent,
        target_branch_percentage=full.branch_coverage_percent,
        greedy_initially_selected_count=len(selected),
        redundancy_elimination_removed_count=0,
        final_selected_scenario_ids=selected,
        removed_scenario_ids=tuple(
            value for value in ("a", "b", "c") if value not in selected
        ),
        coverage_preserved=preserved,
        final_verified_line_identities=full.covered_lines if preserved else (),
        final_verified_branch_identities=(full.covered_branches or ()) if preserved else (),
        final_verified_line_percentage=full.line_coverage_percent if preserved else None,
        final_verified_branch_percentage=full.branch_coverage_percent if preserved else None,
        duration_seconds=0.2,
    )


def _session(
    episodes: tuple[tuple[int, ...], ...],
    *,
    rewards: tuple[float, ...] | None = None,
) -> TrainingSessionResult:
    reward_values = rewards or tuple(10.0 for _ in episodes)
    statistics = tuple(
        EpisodeStatistics(
            episode_number=index,
            step_count=len(indices),
            total_reward=reward_values[index - 1],
            final_coverage_percentage=100.0,
            full_coverage=True,
            executed_test_count=len(indices),
            ordered_action_indices=indices,
            duration_seconds=0.1,
            done_reason="TARGET_COVERAGE_REACHED",
        )
        for index, indices in enumerate(episodes, start=1)
    )
    return TrainingSessionResult(
        episodes=statistics,
        requested_episode_count=len(statistics),
        completed_episode_count=len(statistics),
    )


class _FakeSuite:
    def __init__(
        self,
        tmp_path: Path,
        coverage_by_ids: dict[tuple[str, ...], FunctionCoverageResult],
    ) -> None:
        self.tmp_path = tmp_path
        self.coverage_by_ids = coverage_by_ids
        self.calls: list[tuple[str, ...]] = []

    def measure_scenarios(self, **kwargs: object) -> ScenarioSuiteCoverageResult:
        scenarios = kwargs["scenarios"]
        assert isinstance(scenarios, tuple)
        identifiers = tuple(value.scenario_id for value in scenarios)
        self.calls.append(identifiers)
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        test_file = output / str(kwargs["test_file_name"])
        test_file.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
        coverage = self.coverage_by_ids[identifiers]
        coverage = replace(
            coverage,
            test_file=test_file,
            file_coverage=replace(coverage.file_coverage, test_file=test_file),
        )
        return ScenarioSuiteCoverageResult(
            scenarios=scenarios,
            test_file=test_file,
            coverage=coverage,
        )


def test_service_verifies_exact_identities_and_selects_best_rl_episode(
    tmp_path: Path,
) -> None:
    scenarios = (_scenario("a", 1), _scenario("b", 2), _scenario("c", 3))
    full = _coverage(tmp_path, (1, 2, 3), ((2, -1), (2, 3)))
    missing_arc = _coverage(tmp_path, (1, 2, 3), ((2, 3),))
    fake = _FakeSuite(tmp_path, {("a", "b"): missing_arc, ("a", "c"): full})

    result = StrategyComparisonService(suite_service=fake).compare(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="target",
        function_start_line=1,
        function_end_line=3,
        scenarios=scenarios,
        full_pool_coverage=full,
        greedy_result=_greedy(tmp_path, full, ("a", "b", "c")),
        session_result=_session(((0, 1), (0, 2))),
        output_root=tmp_path / "output",
        timeout_seconds=10.0,
        comparison_timeout_seconds=None,
        rl_hyperparameters=(("epsilon", 0.1),),
        random_seed=42,
    )

    assert result.exact_target_reaching_episode_count == 1
    assert result.best_rl_episode_number == 2
    assert result.winner is StrategyWinner.RL
    assert result.coverage_equality_verified is True
    assert result.rl_verified_line_identities == result.target_line_identities
    assert result.rl_verified_branch_identities == result.target_branch_identities
    assert result.greedy_verified_line_percentage == full.line_coverage_percent
    assert result.greedy_verified_branch_percentage == full.branch_coverage_percent
    assert result.rl_verified_line_percentage == full.line_coverage_percent
    assert result.rl_verified_branch_percentage == full.branch_coverage_percent
    assert result.artifact_paths[0].name == "test_target_rl_selected.py"
    assert result.artifact_paths[0].is_file()
    report = json.loads(result.artifact_paths[1].read_text(encoding="utf-8"))
    assert report["winner"] == "RL"
    assert "keyword_arguments" not in json.dumps(report)


def test_service_caches_duplicate_episode_suite_verification(tmp_path: Path) -> None:
    scenarios = (_scenario("a", 1), _scenario("b", 2), _scenario("c", 3))
    full = _coverage(tmp_path, (1, 2, 3), ((2, -1), (2, 3)))
    fake = _FakeSuite(tmp_path, {("a", "b"): full})

    result = StrategyComparisonService(suite_service=fake).compare(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="target",
        function_start_line=1,
        function_end_line=3,
        scenarios=scenarios,
        full_pool_coverage=full,
        greedy_result=_greedy(tmp_path, full, ("a", "b")),
        session_result=_session(((0, 1), (0, 1)), rewards=(1.0, 2.0)),
        output_root=tmp_path / "output",
        timeout_seconds=10.0,
        comparison_timeout_seconds=None,
        rl_hyperparameters=(),
        random_seed=42,
    )

    assert fake.calls == [("a", "b")]
    assert result.exact_target_reaching_episode_count == 2
    assert result.best_rl_episode_number == 2
    assert result.winner is StrategyWinner.TIE


class _TimeoutSuite(_FakeSuite):
    def measure_scenarios(self, **kwargs: object) -> ScenarioSuiteCoverageResult:
        raise CoverageExecutionTimeoutError("controlled comparison timeout")


def test_service_timeout_preserves_partial_report(tmp_path: Path) -> None:
    scenarios = (_scenario("a", 1), _scenario("b", 2), _scenario("c", 3))
    full = _coverage(tmp_path, (1, 2, 3), ((2, -1), (2, 3)))
    result = StrategyComparisonService(
        suite_service=_TimeoutSuite(tmp_path, {})
    ).compare(
        source_file=tmp_path / "target.py",
        module_path="target",
        function_name="target",
        function_start_line=1,
        function_end_line=3,
        scenarios=scenarios,
        full_pool_coverage=full,
        greedy_result=_greedy(tmp_path, full, ("a", "b")),
        session_result=_session(((0, 1),)),
        output_root=tmp_path / "output",
        timeout_seconds=10.0,
        comparison_timeout_seconds=1.0,
        rl_hyperparameters=(),
        random_seed=42,
    )

    assert result.status is StrategyComparisonStatus.TIMED_OUT
    assert result.episode_traces[0].ordered_scenario_ids == ("a", "b")
    report = result.artifact_paths[0]
    assert json.loads(report.read_text(encoding="utf-8"))["comparison_status"] == "TIMED_OUT"


def test_pending_report_preserves_greedy_before_rl_training(tmp_path: Path) -> None:
    scenarios = (_scenario("a", 1), _scenario("b", 2), _scenario("c", 3))
    full = _coverage(tmp_path, (1, 2, 3), ((2, -1), (2, 3)))

    result = StrategyComparisonService().write_pending(
        source_file=tmp_path / "target.py",
        function_name="target",
        scenarios=scenarios,
        full_pool_coverage=full,
        greedy_result=_greedy(tmp_path, full, ("a", "b")),
        requested_rl_episode_count=3,
        output_root=tmp_path / "output",
        rl_hyperparameters=(("epsilon", 0.2),),
        random_seed=42,
    )

    assert result.status is StrategyComparisonStatus.PARTIAL
    assert result.greedy_selected_scenario_ids == ("a", "b")
    assert result.completed_rl_episode_count == 0
    assert result.artifact_paths[0].is_file()

    progressed = StrategyComparisonService().write_pending(
        source_file=tmp_path / "target.py",
        function_name="target",
        scenarios=scenarios,
        full_pool_coverage=full,
        greedy_result=_greedy(tmp_path, full, ("a", "b")),
        requested_rl_episode_count=3,
        output_root=tmp_path / "output",
        rl_hyperparameters=(("epsilon", 0.2),),
        random_seed=42,
        session_result=_session(((0, 1),)),
    )

    assert progressed.completed_rl_episode_count == 1
    assert progressed.episode_traces[0].ordered_scenario_ids == ("a", "b")


def test_service_propagates_unexpected_runtime_error(tmp_path: Path) -> None:
    scenarios = (_scenario("a", 1), _scenario("b", 2), _scenario("c", 3))
    full = _coverage(tmp_path, (1, 2, 3), ((2, -1), (2, 3)))
    fake = _FakeSuite(tmp_path, {})

    with pytest.raises(KeyError):
        StrategyComparisonService(suite_service=fake).compare(
            source_file=tmp_path / "target.py",
            module_path="target",
            function_name="target",
            function_start_line=1,
            function_end_line=3,
            scenarios=scenarios,
            full_pool_coverage=full,
            greedy_result=_greedy(tmp_path, full, ("a", "b")),
            session_result=_session(((0, 1),)),
            output_root=tmp_path / "output",
            timeout_seconds=10.0,
            comparison_timeout_seconds=None,
            rl_hyperparameters=(),
            random_seed=42,
        )


def test_real_branch_heavy_comparison_preserves_exact_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "branch_target.py"
    source.write_text(
        "def classify(value: int) -> str:\n"
        "    if value < 0:\n"
        "        return 'negative'\n"
        "    if value == 0:\n"
        "        return 'zero'\n"
        "    return 'positive'\n",
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path) if not existing else f"{tmp_path}{os.pathsep}{existing}",
    )

    def scenario(identifier: str, rank: int, value: int, expected: str) -> Scenario:
        return Scenario(
            scenario_id=identifier,
            name=identifier,
            path_index=rank,
            priority_rank=rank,
            priority_level="High",
            dqm_score=100.0 - rank,
            node_ids=(1,),
            edge_labels=(),
            contains_loop=False,
            contains_exception=False,
            description="real branch comparison",
            keyword_arguments=(("value", value),),
            expected_result=expected,
            expected_exception=None,
        )

    scenarios = (
        scenario("negative", 1, -1, "negative"),
        scenario("zero", 2, 0, "zero"),
        scenario("positive", 3, 1, "positive"),
        scenario("duplicate_positive", 4, 2, "positive"),
    )
    suite_service = ScenarioSuiteCoverageService()
    full = suite_service.measure_scenarios(
        source_file=source,
        module_path="branch_target",
        function_name="classify",
        scenarios=scenarios,
        output_directory=tmp_path / "full",
        function_start_line=1,
        function_end_line=6,
        timeout_seconds=20.0,
    ).coverage
    assert isinstance(full, FunctionCoverageResult)
    greedy = ScenarioCoverageMinimizationService().minimize(
        source_file=source,
        module_path="branch_target",
        function_name="classify",
        function_start_line=1,
        function_end_line=6,
        scenarios=scenarios,
        output_root=tmp_path / "output",
        timeout_seconds=20.0,
        full_pool_coverage=full,
    )

    result = StrategyComparisonService().compare(
        source_file=source,
        module_path="branch_target",
        function_name="classify",
        function_start_line=1,
        function_end_line=6,
        scenarios=scenarios,
        full_pool_coverage=full,
        greedy_result=greedy,
        session_result=_session(((0, 1, 2, 3), (0, 1, 2))),
        output_root=tmp_path / "output",
        timeout_seconds=20.0,
        comparison_timeout_seconds=60.0,
        rl_hyperparameters=(("epsilon", 0.0),),
        random_seed=42,
    )

    assert result.coverage_equality_verified is True
    assert result.rl_verified_line_identities == result.target_line_identities
    assert result.rl_verified_branch_identities == result.target_branch_identities
    assert result.greedy_selected_count == 3
    assert result.best_rl_executed_test_count == 3
    assert result.winner is StrategyWinner.TIE
    assert result.exact_target_reaching_episode_count == 2
    assert result.artifact_paths[0].is_relative_to((tmp_path / "output").resolve())
    ast.parse(result.artifact_paths[0].read_text(encoding="utf-8"))
