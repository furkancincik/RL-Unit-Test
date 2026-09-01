from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from models.strategy_evaluation_result import (
    StrategyEvaluationCandidate,
    StrategyEvaluationPool,
    StrategyName,
    StrategyStopReason,
    StrategyWinnerOutcome,
)
from services.strategy_evaluation_service import StrategyEvaluationService
from services.project_coverage_service import ProjectScenarioCoverageSignature


def _candidate(
    identifier: str,
    rank: int,
    *,
    lines: tuple[tuple[str, int], ...] = (),
    branches: tuple[tuple[str, int, int], ...] = (),
    execution_success: bool = True,
) -> StrategyEvaluationCandidate:
    return StrategyEvaluationCandidate(
        candidate_id=identifier,
        dqm_rank=rank,
        covered_line_identities=lines,
        covered_branch_identities=branches,
        execution_success=execution_success,
    )


def _pool() -> StrategyEvaluationPool:
    return StrategyEvaluationPool(
        candidates=(
            _candidate(
                "candidate-a",
                1,
                lines=(("package/module.py", 11),),
                branches=(("package/module.py", 11, 12),),
            ),
            _candidate("candidate-zero", 2),
            _candidate(
                "candidate-b",
                3,
                lines=(("package/module.py", 12),),
            ),
            _candidate(
                "candidate-c",
                4,
                lines=(("package/module.py", 13),),
                branches=(("package/module.py", 13, 15),),
            ),
        ),
        executable_line_identities=(
            ("package/module.py", 11),
            ("package/module.py", 12),
            ("package/module.py", 13),
            ("package/module.py", 99),
        ),
        executable_branch_identities=(
            ("package/module.py", 11, 12),
            ("package/module.py", 13, 15),
            ("package/module.py", 99, 100),
        ),
        target_line_identities=(
            ("package/module.py", 11),
            ("package/module.py", 12),
            ("package/module.py", 13),
        ),
        target_branch_identities=(
            ("package/module.py", 11, 12),
            ("package/module.py", 13, 15),
        ),
    )


def _semantic_result(result: object) -> tuple[object, ...]:
    return (
        result.strategy,
        result.seed,
        result.selected_candidate_ids,
        result.effective_test_count,
        result.zero_contribution_selection_count,
        result.covered_line_identities,
        result.covered_branch_identities,
        result.tests_to_target,
        result.stop_reason,
        tuple(
            (
                step.candidate_id,
                step.new_line_identities,
                step.new_branch_identities,
                step.covered_line_identities,
                step.covered_branch_identities,
            )
            for step in result.coverage_progression
        ),
    )


def test_three_strategies_share_pool_denominator_and_exact_target() -> None:
    report = StrategyEvaluationService().evaluate(
        pool=_pool(),
        rl_seeds=(7, 19),
        rl_episode_count=8,
        epsilon=0.4,
    )

    results = (report.dqm_prefix, report.exact_greedy, *report.rl_runs)
    assert report.common_candidate_ids == (
        "candidate-a",
        "candidate-zero",
        "candidate-b",
        "candidate-c",
    )
    assert all(result.candidate_pool_count == 4 for result in results)
    assert all(result.total_line_count == 4 for result in results)
    assert all(result.total_branch_count == 3 for result in results)
    assert all(result.target_line_identities == _pool().target_line_identities for result in results)
    assert all(result.target_branch_identities == _pool().target_branch_identities for result in results)


def test_project_signature_adapter_preserves_dqm_rank_and_candidate_identity() -> None:
    signatures = tuple(
        ProjectScenarioCoverageSignature(
            project_test_id=candidate.candidate_id,
            original_order=candidate.dqm_rank,
            covered_line_identities=candidate.covered_line_identities,
            covered_branch_identities=candidate.covered_branch_identities,
            execution_success=True,
        )
        for candidate in reversed(_pool().candidates)
    )

    adapted = StrategyEvaluationService.pool_from_project_signatures(
        signatures=signatures,
        executable_line_identities=_pool().executable_line_identities,
        executable_branch_identities=_pool().executable_branch_identities,
        target_line_identities=_pool().target_line_identities,
        target_branch_identities=_pool().target_branch_identities,
    )

    assert adapted == _pool()


def test_dqm_prefix_preserves_rank_and_zero_contribution_is_not_effective() -> None:
    result = StrategyEvaluationService().evaluate_one(
        pool=_pool(),
        strategy=StrategyName.DQM_PREFIX,
    )

    assert result.selected_candidate_ids == (
        "candidate-a",
        "candidate-zero",
        "candidate-b",
        "candidate-c",
    )
    assert result.executed_test_count == 4
    assert result.effective_test_count == 3
    assert result.zero_contribution_selection_count == 1
    assert result.tests_to_target == 4
    assert result.stop_reason is StrategyStopReason.TARGET_REACHED


def test_exact_greedy_uses_exact_marginal_identity_contribution() -> None:
    result = StrategyEvaluationService().evaluate_one(
        pool=_pool(),
        strategy=StrategyName.EXACT_GREEDY,
    )

    assert result.selected_candidate_ids == (
        "candidate-a",
        "candidate-c",
        "candidate-b",
    )
    assert tuple(step.exact_new_identity_count for step in result.coverage_progression) == (2, 2, 1)
    assert result.zero_contribution_selection_count == 0
    assert result.tests_to_target == 3


def test_concrete_rejected_or_failed_candidate_cannot_enter_common_pool() -> None:
    with pytest.raises(ValueError, match="execution_success"):
        StrategyEvaluationPool(
            candidates=(
                _candidate(
                    "rejected",
                    1,
                    lines=(("unit.py", 1),),
                    execution_success=False,
                ),
            ),
            executable_line_identities=(("unit.py", 1),),
            executable_branch_identities=(),
            target_line_identities=(("unit.py", 1),),
            target_branch_identities=(),
        )


def test_rl_seed_is_reproducible_without_global_random_state() -> None:
    service = StrategyEvaluationService()
    first = service.evaluate_one(
        pool=_pool(),
        strategy=StrategyName.RL,
        seed=31,
        rl_episode_count=12,
        epsilon=0.6,
    )
    second = service.evaluate_one(
        pool=_pool(),
        strategy=StrategyName.RL,
        seed=31,
        rl_episode_count=12,
        epsilon=0.6,
    )

    assert _semantic_result(first) == _semantic_result(second)
    assert first.q_state_count == second.q_state_count
    assert first.actions_evaluated == second.actions_evaluated
    assert first.exploration_selection_count == second.exploration_selection_count
    assert first.exploitation_selection_count == second.exploitation_selection_count


def test_strategy_evaluation_order_does_not_leak_state() -> None:
    service = StrategyEvaluationService()
    forward = {
        strategy: service.evaluate_one(
            pool=_pool(),
            strategy=strategy,
            seed=13 if strategy is StrategyName.RL else None,
            rl_episode_count=10,
            epsilon=0.5,
        )
        for strategy in (
            StrategyName.DQM_PREFIX,
            StrategyName.EXACT_GREEDY,
            StrategyName.RL,
        )
    }
    reverse = {
        strategy: service.evaluate_one(
            pool=_pool(),
            strategy=strategy,
            seed=13 if strategy is StrategyName.RL else None,
            rl_episode_count=10,
            epsilon=0.5,
        )
        for strategy in (
            StrategyName.RL,
            StrategyName.EXACT_GREEDY,
            StrategyName.DQM_PREFIX,
        )
    }

    assert {
        strategy: _semantic_result(result) for strategy, result in forward.items()
    } == {
        strategy: _semantic_result(result) for strategy, result in reverse.items()
    }


def test_unreachable_target_has_controlled_null_tests_to_target() -> None:
    pool = StrategyEvaluationPool(
        candidates=(
            _candidate("only", 1, lines=(("unit.py", 1),)),
        ),
        executable_line_identities=(("unit.py", 1), ("unit.py", 2)),
        executable_branch_identities=(),
        target_line_identities=(("unit.py", 1), ("unit.py", 2)),
        target_branch_identities=(),
    )

    for strategy in StrategyName:
        result = StrategyEvaluationService().evaluate_one(
            pool=pool,
            strategy=strategy,
            seed=5 if strategy is StrategyName.RL else None,
            rl_episode_count=3,
        )
        assert result.tests_to_target is None
        assert result.stop_reason in {
            StrategyStopReason.CANDIDATES_EXHAUSTED,
            StrategyStopReason.NO_FURTHER_CONTRIBUTION,
        }


def test_public_summary_contains_only_safe_aggregate_and_exact_identity_data(
    tmp_path: Path,
) -> None:
    service = StrategyEvaluationService()
    report = service.evaluate(
        pool=_pool(),
        rl_seeds=(2,),
        rl_episode_count=4,
    )
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert "raw_source" not in serialized
    assert "keyword_arguments" not in serialized
    assert "setup_plan" not in serialized
    assert "q_table" not in serialized
    assert "traceback" not in serialized
    assert "memory_address" not in serialized
    assert payload["globally_minimal"] is False
    report_path = service.write_report(report, output_root=tmp_path)
    persisted = report_path.read_text(encoding="utf-8")
    assert json.loads(persisted) == payload
    assert str(tmp_path.resolve()) not in persisted


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("executed_test_count", -1),
    ),
)
def test_result_model_rejects_non_finite_or_negative_metrics(field: str, value: object) -> None:
    result = StrategyEvaluationService().evaluate_one(
        pool=_pool(),
        strategy=StrategyName.DQM_PREFIX,
    )
    values = {
        name: getattr(result, name)
        for name in result.__dataclass_fields__
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        type(result)(**values)


def test_winner_contract_is_lexicographic_and_reports_mixed_seeds() -> None:
    service = StrategyEvaluationService()
    dqm = service.evaluate_one(pool=_pool(), strategy=StrategyName.DQM_PREFIX)
    exact = service.evaluate_one(pool=_pool(), strategy=StrategyName.EXACT_GREEDY)
    slower_baseline = replace(
        dqm,
        strategy=StrategyName.EXACT_GREEDY,
        duration_seconds=dqm.duration_seconds,
    )
    winning_rl = replace(
        exact,
        strategy=StrategyName.RL,
        seed=1,
    )
    losing_rl = replace(
        dqm,
        strategy=StrategyName.RL,
        seed=2,
        duration_seconds=dqm.duration_seconds + 1.0,
    )

    assert service.determine_winner(
        dqm=dqm,
        greedy=slower_baseline,
        rl_runs=(winning_rl,),
    ) is StrategyWinnerOutcome.RL_WINS
    assert service.determine_winner(
        dqm=dqm,
        greedy=exact,
        rl_runs=(losing_rl,),
    ) is StrategyWinnerOutcome.BASELINE_WINS
    assert service.determine_winner(
        dqm=dqm,
        greedy=slower_baseline,
        rl_runs=(
            replace(
                slower_baseline,
                strategy=StrategyName.RL,
                seed=3,
            ),
        ),
    ) is StrategyWinnerOutcome.TIE
    assert service.determine_winner(
        dqm=dqm,
        greedy=slower_baseline,
        rl_runs=(winning_rl, losing_rl),
    ) is StrategyWinnerOutcome.MIXED
