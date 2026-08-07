from __future__ import annotations

from unittest.mock import Mock

import pytest

from rl.action import Action
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_learning_trainer import QLearningTrainer
from rl.q_table import QTable
from rl.state_encoder import StateEncoder
from rl.training_session import (
    TrainingSession,
    TrainingSessionResult,
)
from rl.training_statistics import (
    EpisodeStatistics,
    TrainingStatistics,
)


def create_initial_state() -> CoverageState:
    """Testlerde kullanılacak başlangıç coverage durumunu oluşturur."""
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4),
        uncovered_branches=2,
    )


def create_environment() -> CoverageEnvironment:
    """
    Her episode içerisinde tek adımda tam coverage sağlayan
    kontrollü CoverageEnvironment oluşturur.
    """
    initial_state = create_initial_state()

    def transition_function(
        state: CoverageState,
        action: Action,
    ) -> CoverageState:
        return CoverageState(
            coverage_percentage=100.0,
            executed_tests=state.executed_tests + 1,
            missing_lines=(),
            uncovered_branches=0,
        )

    return CoverageEnvironment(
        initial_state=initial_state,
        actions=(
            Action(scenario_index=0),
        ),
        transition_function=transition_function,
    )


def create_completed_steps(
    *,
    reward: float = 109.9,
    coverage_percentage: float = 100.0,
    executed_tests: int = 1,
) -> tuple[EnvironmentStep, ...]:
    """Tamamlanmış tek adımlı episode sonucu oluşturur."""
    return (
        EnvironmentStep(
            state=CoverageState(
                coverage_percentage=coverage_percentage,
                executed_tests=executed_tests,
                missing_lines=(
                    () if coverage_percentage == 100.0 else (1,)
                ),
                uncovered_branches=(
                    0 if coverage_percentage == 100.0 else 1
                ),
            ),
            reward=reward,
            done=True,
        ),
    )


def create_mock_trainer(
    *,
    side_effect: list[tuple[EnvironmentStep, ...]] | None = None,
) -> Mock:
    """Kontrollü QLearningTrainer mock nesnesi oluşturur."""
    trainer = Mock(
        spec=QLearningTrainer,
    )

    if side_effect is None:
        trainer.train_episode.return_value = (
            create_completed_steps()
        )
    else:
        trainer.train_episode.side_effect = side_effect

    return trainer


def create_real_trainer_with_epsilon(
    epsilon: float,
) -> QLearningTrainer:
    """Epsilon decay testleri için gerçek trainer zinciri oluşturur."""
    agent = QLearningAgent(
        q_table=QTable(),
        policy=EpsilonGreedyPolicy(
            epsilon=epsilon,
        ),
        learning_rate=0.5,
        discount_factor=0.9,
    )

    return QLearningTrainer(
        agent=agent,
        state_encoder=StateEncoder(),
    )


def test_session_uses_supplied_trainer() -> None:
    trainer = create_mock_trainer()

    session = TrainingSession(
        trainer=trainer,
    )

    assert session.trainer is trainer


def test_session_creates_default_statistics() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    assert isinstance(
        session.statistics,
        TrainingStatistics,
    )


def test_session_uses_supplied_statistics() -> None:
    statistics = TrainingStatistics()

    session = TrainingSession(
        trainer=create_mock_trainer(),
        statistics=statistics,
    )

    assert session.statistics is statistics


def test_run_returns_training_session_result() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    result = session.run(
        environment=create_environment(),
        episode_count=2,
    )

    assert isinstance(
        result,
        TrainingSessionResult,
    )


def test_run_executes_requested_episode_count() -> None:
    trainer = create_mock_trainer()

    session = TrainingSession(
        trainer=trainer,
    )

    session.run(
        environment=create_environment(),
        episode_count=3,
    )

    assert trainer.train_episode.call_count == 3


def test_run_resets_environment_for_each_episode() -> None:
    trainer = create_mock_trainer()

    session = TrainingSession(
        trainer=trainer,
    )

    environment = create_environment()

    session.run(
        environment=environment,
        episode_count=2,
    )

    trainer.train_episode.assert_any_call(
        environment=environment,
        reset=True,
    )

    assert all(
        call.kwargs["reset"] is True
        for call in trainer.train_episode.call_args_list
    )


def test_run_records_each_episode() -> None:
    trainer = create_mock_trainer(
        side_effect=[
            create_completed_steps(
                reward=50.0,
            ),
            create_completed_steps(
                reward=75.0,
            ),
        ]
    )

    session = TrainingSession(
        trainer=trainer,
    )

    result = session.run(
        environment=create_environment(),
        episode_count=2,
    )

    assert len(result.episodes) == 2
    assert result.episodes[0].total_reward == 50.0
    assert result.episodes[1].total_reward == 75.0


def test_run_assigns_sequential_episode_numbers() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    result = session.run(
        environment=create_environment(),
        episode_count=3,
    )

    assert tuple(
        episode.episode_number
        for episode in result.episodes
    ) == (1, 2, 3)


def test_run_returns_requested_and_completed_counts() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    result = session.run(
        environment=create_environment(),
        episode_count=4,
    )

    assert result.requested_episode_count == 4
    assert result.completed_episode_count == 4


def test_result_success_when_all_episodes_complete() -> None:
    result = TrainingSessionResult(
        episodes=(
            EpisodeStatistics(
                episode_number=1,
                step_count=1,
                total_reward=10.0,
                final_coverage_percentage=100.0,
                full_coverage=True,
                executed_test_count=1,
            ),
        ),
        requested_episode_count=1,
        completed_episode_count=1,
    )

    assert result.success is True


def test_result_not_successful_when_counts_differ() -> None:
    result = TrainingSessionResult(
        episodes=(),
        requested_episode_count=2,
        completed_episode_count=1,
    )

    assert result.success is False


def test_result_counts_full_coverage_episodes() -> None:
    result = TrainingSessionResult(
        episodes=(
            EpisodeStatistics(
                episode_number=1,
                step_count=2,
                total_reward=50.0,
                final_coverage_percentage=100.0,
                full_coverage=True,
                executed_test_count=2,
            ),
            EpisodeStatistics(
                episode_number=2,
                step_count=3,
                total_reward=40.0,
                final_coverage_percentage=90.0,
                full_coverage=False,
                executed_test_count=3,
            ),
            EpisodeStatistics(
                episode_number=3,
                step_count=1,
                total_reward=60.0,
                final_coverage_percentage=100.0,
                full_coverage=True,
                executed_test_count=1,
            ),
        ),
        requested_episode_count=3,
        completed_episode_count=3,
    )

    assert result.full_coverage_episode_count == 2


def test_best_episode_prefers_fewer_executed_tests_at_same_coverage() -> None:
    slow_episode = EpisodeStatistics(
        episode_number=1,
        step_count=3,
        total_reward=120.0,
        final_coverage_percentage=100.0,
        full_coverage=True,
        executed_test_count=3,
    )

    fast_episode = EpisodeStatistics(
        episode_number=2,
        step_count=2,
        total_reward=90.0,
        final_coverage_percentage=100.0,
        full_coverage=True,
        executed_test_count=2,
    )

    result = TrainingSessionResult(
        episodes=(
            slow_episode,
            fast_episode,
        ),
        requested_episode_count=2,
        completed_episode_count=2,
    )

    assert result.best_episode is fast_episode


def test_best_episode_prefers_higher_reward_when_steps_equal() -> None:
    first_episode = EpisodeStatistics(
        episode_number=1,
        step_count=2,
        total_reward=70.0,
        final_coverage_percentage=100.0,
        full_coverage=True,
        executed_test_count=2,
    )

    second_episode = EpisodeStatistics(
        episode_number=2,
        step_count=2,
        total_reward=90.0,
        final_coverage_percentage=100.0,
        full_coverage=True,
        executed_test_count=2,
    )

    result = TrainingSessionResult(
        episodes=(
            first_episode,
            second_episode,
        ),
        requested_episode_count=2,
        completed_episode_count=2,
    )

    assert result.best_episode is second_episode


def test_best_episode_returns_highest_coverage_without_full_coverage() -> None:
    lower_coverage_episode = EpisodeStatistics(
        episode_number=1,
        step_count=2,
        total_reward=80.0,
        final_coverage_percentage=80.0,
        full_coverage=False,
        executed_test_count=2,
    )

    higher_coverage_episode = EpisodeStatistics(
        episode_number=2,
        step_count=3,
        total_reward=40.0,
        final_coverage_percentage=90.0,
        full_coverage=False,
        executed_test_count=3,
    )

    result = TrainingSessionResult(
        episodes=(
            lower_coverage_episode,
            higher_coverage_episode,
        ),
        requested_episode_count=2,
        completed_episode_count=2,
    )

    assert result.best_episode is higher_coverage_episode


def test_run_clears_previous_statistics_by_default() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_steps(
            reward=10.0,
        ),
    )

    session = TrainingSession(
        trainer=create_mock_trainer(),
        statistics=statistics,
    )

    result = session.run(
        environment=create_environment(),
        episode_count=1,
    )

    assert len(result.episodes) == 1
    assert session.statistics.episode_count == 1
    assert session.statistics.episodes[0].episode_number == 1


def test_run_preserves_statistics_when_clear_disabled() -> None:
    statistics = TrainingStatistics()

    first_episode = statistics.record_episode(
        steps=create_completed_steps(
            reward=10.0,
        ),
    )

    session = TrainingSession(
        trainer=create_mock_trainer(),
        statistics=statistics,
    )

    result = session.run(
        environment=create_environment(),
        episode_count=1,
        clear_statistics=False,
    )

    assert result.episodes[0].episode_number == 2
    assert session.statistics.episode_count == 2
    assert session.statistics.episodes[0] is first_episode


def test_run_updates_statistics_summary() -> None:
    trainer = create_mock_trainer(
        side_effect=[
            create_completed_steps(
                reward=50.0,
            ),
            create_completed_steps(
                reward=80.0,
            ),
        ]
    )

    session = TrainingSession(
        trainer=trainer,
    )

    session.run(
        environment=create_environment(),
        episode_count=2,
    )

    assert session.statistics.total_reward == 130.0
    assert session.statistics.average_reward == 65.0
    assert session.statistics.best_reward == 80.0
    assert session.statistics.full_coverage_episode_count == 2


def test_session_rejects_invalid_trainer() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "trainer bir QLearningTrainer "
            "örneği olmalıdır"
        ),
    ):
        TrainingSession(
            trainer="invalid",  # type: ignore[arg-type]
        )


def test_session_rejects_invalid_statistics() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "statistics bir TrainingStatistics "
            "örneği veya None olmalıdır"
        ),
    ):
        TrainingSession(
            trainer=create_mock_trainer(),
            statistics="invalid",  # type: ignore[arg-type]
        )


def test_run_rejects_invalid_environment() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "environment bir CoverageEnvironment "
            "örneği olmalıdır"
        ),
    ):
        session.run(
            environment="invalid",  # type: ignore[arg-type]
            episode_count=1,
        )


def test_run_rejects_non_integer_episode_count() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        TypeError,
        match="episode_count bir tam sayı olmalıdır",
    ):
        session.run(
            environment=create_environment(),
            episode_count=1.5,  # type: ignore[arg-type]
        )


def test_run_rejects_boolean_episode_count() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        TypeError,
        match="episode_count bir tam sayı olmalıdır",
    ):
        session.run(
            environment=create_environment(),
            episode_count=True,  # type: ignore[arg-type]
        )


def test_run_rejects_non_positive_episode_count() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "episode_count 1 veya "
            "daha büyük olmalıdır"
        ),
    ):
        session.run(
            environment=create_environment(),
            episode_count=0,
        )


def test_run_rejects_invalid_clear_statistics() -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        TypeError,
        match="clear_statistics bool olmalıdır",
    ):
        session.run(
            environment=create_environment(),
            episode_count=1,
            clear_statistics="invalid",  # type: ignore[arg-type]
        )


def test_best_episode_prioritizes_coverage_over_test_count() -> None:
    fewer_tests_lower_coverage = EpisodeStatistics(
        episode_number=1,
        step_count=1,
        total_reward=200.0,
        final_coverage_percentage=90.0,
        full_coverage=False,
        executed_test_count=1,
    )

    more_tests_higher_coverage = EpisodeStatistics(
        episode_number=2,
        step_count=4,
        total_reward=50.0,
        final_coverage_percentage=95.0,
        full_coverage=False,
        executed_test_count=4,
    )

    result = TrainingSessionResult(
        episodes=(
            fewer_tests_lower_coverage,
            more_tests_higher_coverage,
        ),
        requested_episode_count=2,
        completed_episode_count=2,
    )

    assert result.best_episode is more_tests_higher_coverage


def test_best_episode_prefers_fewer_tests_when_coverage_equal() -> None:
    more_tests_episode = EpisodeStatistics(
        episode_number=1,
        step_count=6,
        total_reward=100.0,
        final_coverage_percentage=95.0,
        full_coverage=False,
        executed_test_count=6,
    )

    fewer_tests_episode = EpisodeStatistics(
        episode_number=2,
        step_count=3,
        total_reward=70.0,
        final_coverage_percentage=95.0,
        full_coverage=False,
        executed_test_count=3,
    )

    result = TrainingSessionResult(
        episodes=(
            more_tests_episode,
            fewer_tests_episode,
        ),
        requested_episode_count=2,
        completed_episode_count=2,
    )

    assert result.best_episode is fewer_tests_episode


def test_best_episode_prefers_earlier_episode_when_all_metrics_equal() -> None:
    first_episode = EpisodeStatistics(
        episode_number=1,
        step_count=3,
        total_reward=80.0,
        final_coverage_percentage=95.0,
        full_coverage=False,
        executed_test_count=3,
    )

    second_episode = EpisodeStatistics(
        episode_number=2,
        step_count=3,
        total_reward=80.0,
        final_coverage_percentage=95.0,
        full_coverage=False,
        executed_test_count=3,
    )

    result = TrainingSessionResult(
        episodes=(
            first_episode,
            second_episode,
        ),
        requested_episode_count=2,
        completed_episode_count=2,
    )

    assert result.best_episode is first_episode


def test_best_episode_returns_none_when_no_episodes_exist() -> None:
    result = TrainingSessionResult(
        episodes=(),
        requested_episode_count=1,
        completed_episode_count=0,
    )

    assert result.best_episode is None


def test_run_does_not_change_epsilon_when_decay_is_disabled() -> None:
    trainer = create_real_trainer_with_epsilon(
        0.30
    )

    session = TrainingSession(
        trainer=trainer,
    )

    session.run(
        environment=create_environment(),
        episode_count=2,
    )

    assert (
        trainer.agent.policy.epsilon
        == pytest.approx(0.30)
    )


def test_run_decays_epsilon_after_each_completed_episode() -> None:
    trainer = create_real_trainer_with_epsilon(
        0.40
    )

    session = TrainingSession(
        trainer=trainer,
    )

    session.run(
        environment=create_environment(),
        episode_count=3,
        epsilon_decay_rate=0.5,
        minimum_epsilon=0.05,
    )

    assert (
        trainer.agent.policy.epsilon
        == pytest.approx(0.05)
    )


def test_run_respects_minimum_epsilon_during_decay() -> None:
    trainer = create_real_trainer_with_epsilon(
        0.20
    )

    session = TrainingSession(
        trainer=trainer,
    )

    session.run(
        environment=create_environment(),
        episode_count=5,
        epsilon_decay_rate=0.5,
        minimum_epsilon=0.05,
    )

    assert (
        trainer.agent.policy.epsilon
        == pytest.approx(0.05)
    )


def test_run_preserves_episode_results_when_epsilon_decay_is_enabled() -> None:
    trainer = create_real_trainer_with_epsilon(
        0.20
    )

    session = TrainingSession(
        trainer=trainer,
    )

    result = session.run(
        environment=create_environment(),
        episode_count=2,
        epsilon_decay_rate=0.9,
        minimum_epsilon=0.05,
    )

    assert result.success is True
    assert len(result.episodes) == 2
    assert (
        result.completed_episode_count
        == 2
    )


@pytest.mark.parametrize(
    "epsilon_decay_rate",
    (
        -0.1,
        1.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_run_rejects_invalid_epsilon_decay_rate(
    epsilon_decay_rate: float,
) -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        ValueError,
    ):
        session.run(
            environment=create_environment(),
            episode_count=1,
            epsilon_decay_rate=epsilon_decay_rate,
        )


@pytest.mark.parametrize(
    "epsilon_decay_rate",
    (
        True,
        "0.9",
        object(),
    ),
)
def test_run_rejects_invalid_epsilon_decay_rate_type(
    epsilon_decay_rate: object,
) -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "epsilon_decay_rate sayısal bir değer "
            "veya None olmalıdır."
        ),
    ):
        session.run(
            environment=create_environment(),
            episode_count=1,
            epsilon_decay_rate=(  # type: ignore[arg-type]
                epsilon_decay_rate
            ),
        )


@pytest.mark.parametrize(
    "minimum_epsilon",
    (
        -0.1,
        1.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_run_rejects_invalid_minimum_epsilon(
    minimum_epsilon: float,
) -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        ValueError,
    ):
        session.run(
            environment=create_environment(),
            episode_count=1,
            minimum_epsilon=minimum_epsilon,
        )


@pytest.mark.parametrize(
    "minimum_epsilon",
    (
        True,
        "0.05",
        object(),
    ),
)
def test_run_rejects_invalid_minimum_epsilon_type(
    minimum_epsilon: object,
) -> None:
    session = TrainingSession(
        trainer=create_mock_trainer(),
    )

    with pytest.raises(
        TypeError,
        match="minimum_epsilon sayısal olmalıdır.",
    ):
        session.run(
            environment=create_environment(),
            episode_count=1,
            minimum_epsilon=(  # type: ignore[arg-type]
                minimum_epsilon
            ),
        )


def test_run_propagates_error_when_minimum_exceeds_current_epsilon() -> None:
    trainer = create_real_trainer_with_epsilon(
        0.10
    )

    session = TrainingSession(
        trainer=trainer,
    )

    with pytest.raises(
        ValueError,
        match=(
            "minimum_epsilon cannot be greater "
            "than current epsilon."
        ),
    ):
        session.run(
            environment=create_environment(),
            episode_count=1,
            epsilon_decay_rate=0.9,
            minimum_epsilon=0.20,
        )

