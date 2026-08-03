from __future__ import annotations

import pytest

from rl.coverage_state import CoverageState
from rl.environment_step import EnvironmentStep
from rl.training_statistics import (
    EpisodeStatistics,
    TrainingStatistics,
)


def create_state(
    *,
    coverage_percentage: float,
    executed_tests: int,
    missing_lines: tuple[int, ...],
    uncovered_branches: int,
) -> CoverageState:
    """Testlerde kullanılacak CoverageState nesnesini oluşturur."""
    return CoverageState(
        coverage_percentage=coverage_percentage,
        executed_tests=executed_tests,
        missing_lines=missing_lines,
        uncovered_branches=uncovered_branches,
    )


def create_step(
    *,
    coverage_percentage: float,
    executed_tests: int,
    reward: float,
    done: bool,
    missing_lines: tuple[int, ...] = (),
    uncovered_branches: int = 0,
) -> EnvironmentStep:
    """Kontrollü EnvironmentStep nesnesi oluşturur."""
    return EnvironmentStep(
        state=create_state(
            coverage_percentage=coverage_percentage,
            executed_tests=executed_tests,
            missing_lines=missing_lines,
            uncovered_branches=uncovered_branches,
        ),
        reward=reward,
        done=done,
    )


def create_completed_episode(
    *,
    rewards: tuple[float, ...] = (
        24.9,
        34.9,
        49.9,
    ),
    full_coverage: bool = True,
) -> tuple[EnvironmentStep, ...]:
    """Tamamlanmış örnek episode adımları oluşturur."""
    if len(rewards) != 3:
        raise ValueError(
            "Test yardımcısı tam olarak üç reward bekler."
        )

    return (
        create_step(
            coverage_percentage=40.0,
            executed_tests=1,
            reward=rewards[0],
            done=False,
            missing_lines=(1, 2, 3),
            uncovered_branches=3,
        ),
        create_step(
            coverage_percentage=75.0,
            executed_tests=2,
            reward=rewards[1],
            done=False,
            missing_lines=(1,),
            uncovered_branches=1,
        ),
        create_step(
            coverage_percentage=(
                100.0 if full_coverage else 90.0
            ),
            executed_tests=3,
            reward=rewards[2],
            done=True,
            missing_lines=(
                () if full_coverage else (1,)
            ),
            uncovered_branches=(
                0 if full_coverage else 1
            ),
        ),
    )


def test_record_episode_returns_episode_statistics() -> None:
    statistics = TrainingStatistics()

    result = statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert isinstance(
        result,
        EpisodeStatistics,
    )


def test_record_episode_assigns_sequential_number() -> None:
    statistics = TrainingStatistics()

    first = statistics.record_episode(
        steps=create_completed_episode(),
    )
    second = statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert first.episode_number == 1
    assert second.episode_number == 2


def test_record_episode_calculates_step_count() -> None:
    statistics = TrainingStatistics()

    result = statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert result.step_count == 3


def test_record_episode_calculates_total_reward() -> None:
    statistics = TrainingStatistics()

    result = statistics.record_episode(
        steps=create_completed_episode(
            rewards=(10.0, 20.0, 30.0),
        ),
    )

    assert result.total_reward == 60.0


def test_record_episode_uses_final_coverage() -> None:
    statistics = TrainingStatistics()

    result = statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert result.final_coverage_percentage == 100.0
    assert result.full_coverage is True


def test_record_episode_uses_final_executed_test_count() -> None:
    statistics = TrainingStatistics()

    result = statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert result.executed_test_count == 3


def test_episodes_property_returns_recorded_results() -> None:
    statistics = TrainingStatistics()

    episode = statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert statistics.episodes == (
        episode,
    )


def test_episode_count_returns_record_count() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(),
    )
    statistics.record_episode(
        steps=create_completed_episode(),
    )

    assert statistics.episode_count == 2


def test_total_reward_sums_all_episodes() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            rewards=(10.0, 10.0, 10.0),
        ),
    )
    statistics.record_episode(
        steps=create_completed_episode(
            rewards=(20.0, 20.0, 20.0),
        ),
    )

    assert statistics.total_reward == 90.0


def test_average_reward_calculates_episode_average() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            rewards=(10.0, 10.0, 10.0),
        ),
    )
    statistics.record_episode(
        steps=create_completed_episode(
            rewards=(20.0, 20.0, 20.0),
        ),
    )

    assert statistics.average_reward == 45.0


def test_best_reward_returns_highest_episode_reward() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            rewards=(10.0, 10.0, 10.0),
        ),
    )
    statistics.record_episode(
        steps=create_completed_episode(
            rewards=(30.0, 30.0, 30.0),
        ),
    )

    assert statistics.best_reward == 90.0


def test_average_step_count_returns_average() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(),
    )
    statistics.record_episode(
        steps=(
            create_step(
                coverage_percentage=60.0,
                executed_tests=1,
                reward=20.0,
                done=False,
                missing_lines=(1,),
                uncovered_branches=1,
            ),
            create_step(
                coverage_percentage=100.0,
                executed_tests=2,
                reward=50.0,
                done=True,
            ),
        ),
    )

    assert statistics.average_step_count == 2.5


def test_best_step_count_uses_only_full_coverage_episodes() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=False,
        ),
    )

    statistics.record_episode(
        steps=(
            create_step(
                coverage_percentage=60.0,
                executed_tests=1,
                reward=20.0,
                done=False,
                missing_lines=(1,),
                uncovered_branches=1,
            ),
            create_step(
                coverage_percentage=100.0,
                executed_tests=2,
                reward=50.0,
                done=True,
            ),
        ),
    )

    assert statistics.best_step_count == 2


def test_best_step_count_returns_zero_without_full_coverage() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=False,
        ),
    )

    assert statistics.best_step_count == 0


def test_best_coverage_percentage_returns_highest_value() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=False,
        ),
    )
    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=True,
        ),
    )

    assert statistics.best_coverage_percentage == 100.0


def test_full_coverage_episode_count() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=True,
        ),
    )
    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=False,
        ),
    )
    statistics.record_episode(
        steps=create_completed_episode(
            full_coverage=True,
        ),
    )

    assert statistics.full_coverage_episode_count == 2


def test_clear_removes_all_statistics() -> None:
    statistics = TrainingStatistics()

    statistics.record_episode(
        steps=create_completed_episode(),
    )

    statistics.clear()

    assert statistics.episodes == ()
    assert statistics.episode_count == 0
    assert statistics.total_reward == 0.0
    assert statistics.average_reward == 0.0
    assert statistics.best_reward == 0.0
    assert statistics.average_step_count == 0.0
    assert statistics.best_step_count == 0
    assert statistics.best_coverage_percentage == 0.0
    assert statistics.full_coverage_episode_count == 0


def test_record_episode_rejects_non_tuple_steps() -> None:
    statistics = TrainingStatistics()

    with pytest.raises(
        TypeError,
        match=(
            "steps bir EnvironmentStep tuple'ı "
            "olmalıdır"
        ),
    ):
        statistics.record_episode(
            steps=[],  # type: ignore[arg-type]
        )


def test_record_episode_rejects_empty_steps() -> None:
    statistics = TrainingStatistics()

    with pytest.raises(
        ValueError,
        match="Episode en az bir adım içermelidir",
    ):
        statistics.record_episode(
            steps=(),
        )


def test_record_episode_rejects_invalid_step_item() -> None:
    statistics = TrainingStatistics()

    with pytest.raises(
        TypeError,
        match=(
            "steps yalnızca EnvironmentStep "
            "nesneleri içermelidir"
        ),
    ):
        statistics.record_episode(
            steps=(
                "invalid",  # type: ignore[arg-type]
            ),
        )


def test_record_episode_rejects_incomplete_episode() -> None:
    statistics = TrainingStatistics()

    steps = (
        create_step(
            coverage_percentage=50.0,
            executed_tests=1,
            reward=20.0,
            done=False,
            missing_lines=(1,),
            uncovered_branches=1,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Episode tamamlanmadan "
            "istatistik kaydedilemez"
        ),
    ):
        statistics.record_episode(
            steps=steps,
        )


def test_episode_statistics_rejects_invalid_episode_number() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "episode_number 1 veya "
            "daha büyük olmalıdır"
        ),
    ):
        EpisodeStatistics(
            episode_number=0,
            step_count=1,
            total_reward=10.0,
            final_coverage_percentage=100.0,
            full_coverage=True,
            executed_test_count=1,
        )


def test_episode_statistics_rejects_invalid_coverage() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "final_coverage_percentage "
            "0 ile 100 arasında olmalıdır"
        ),
    ):
        EpisodeStatistics(
            episode_number=1,
            step_count=1,
            total_reward=10.0,
            final_coverage_percentage=101.0,
            full_coverage=True,
            executed_test_count=1,
        )


def test_episode_statistics_rejects_non_boolean_full_coverage() -> None:
    with pytest.raises(
        TypeError,
        match="full_coverage bool olmalıdır",
    ):
        EpisodeStatistics(
            episode_number=1,
            step_count=1,
            total_reward=10.0,
            final_coverage_percentage=100.0,
            full_coverage=1,  # type: ignore[arg-type]
            executed_test_count=1,
        )