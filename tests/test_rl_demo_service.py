from __future__ import annotations

from services.rl_demo_service import (
    RLDemoService,
    RLDemoStep,
    RLDemoSummary,
)


def test_run_returns_demo_summary() -> None:
    service = RLDemoService()

    summary = service.run()

    assert isinstance(
        summary,
        RLDemoSummary,
    )


def test_run_starts_from_zero_coverage() -> None:
    service = RLDemoService()

    summary = service.run()

    assert (
        summary.initial_state.coverage_percentage
        == 0.0
    )
    assert summary.initial_state.executed_tests == 0
    assert summary.initial_state.missing_line_count == 6
    assert summary.initial_state.uncovered_branches == 4


def test_run_reaches_full_coverage() -> None:
    service = RLDemoService()

    summary = service.run()

    assert (
        summary.final_state.coverage_percentage
        == 100.0
    )
    assert summary.final_state.missing_lines == ()
    assert summary.final_state.uncovered_branches == 0
    assert summary.completed is True


def test_run_executes_three_training_steps() -> None:
    service = RLDemoService()

    summary = service.run()

    assert summary.step_count == 3
    assert len(summary.steps) == 3

    assert all(
        isinstance(step, RLDemoStep)
        for step in summary.steps
    )


def test_run_uses_each_action_once() -> None:
    service = RLDemoService()

    summary = service.run()

    selected_indices = {
        step.selected_action.scenario_index
        for step in summary.steps
    }

    assert selected_indices == {
        0,
        1,
        2,
    }


def test_run_increases_executed_test_count() -> None:
    service = RLDemoService()

    summary = service.run()

    executed_test_counts = [
        step.next_state.executed_tests
        for step in summary.steps
    ]

    assert executed_test_counts == [
        1,
        2,
        3,
    ]


def test_run_increases_coverage_after_each_action() -> None:
    service = RLDemoService()

    summary = service.run()

    previous_coverage = (
        summary.initial_state.coverage_percentage
    )

    for step in summary.steps:
        assert (
            step.next_state.coverage_percentage
            > previous_coverage
        )

        previous_coverage = (
            step.next_state.coverage_percentage
        )


def test_run_updates_q_value_for_each_step() -> None:
    service = RLDemoService()

    summary = service.run()

    for step in summary.steps:
        assert step.old_q_value == 0.0
        assert step.new_q_value != step.old_q_value


def test_run_encodes_previous_and_next_states() -> None:
    service = RLDemoService()

    summary = service.run()

    for step in summary.steps:
        assert step.previous_state_key is not None
        assert step.next_state_key is not None


def test_run_marks_only_last_step_as_completed() -> None:
    service = RLDemoService()

    summary = service.run()

    assert summary.steps[0].done is False
    assert summary.steps[1].done is False
    assert summary.steps[2].done is True


def test_run_calculates_total_reward() -> None:
    service = RLDemoService()

    summary = service.run()

    expected_total_reward = round(
        sum(
            step.reward
            for step in summary.steps
        ),
        2,
    )

    assert summary.total_reward == expected_total_reward


def test_run_is_deterministic() -> None:
    """
    Sabit random seed nedeniyle demo her çalıştırıldığında
    aynı aksiyon sırasını üretmelidir.
    """
    first_summary = RLDemoService().run()
    second_summary = RLDemoService().run()

    first_action_order = [
        step.selected_action.scenario_index
        for step in first_summary.steps
    ]

    second_action_order = [
        step.selected_action.scenario_index
        for step in second_summary.steps
    ]

    assert first_action_order == second_action_order