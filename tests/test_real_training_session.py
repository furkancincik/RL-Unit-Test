from __future__ import annotations

import random
from pathlib import Path

from generator.scenario_generator import Scenario
from rl.coverage_environment import CoverageEnvironment
from rl.coverage_state import CoverageState
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_learning_agent import QLearningAgent
from rl.q_learning_trainer import QLearningTrainer
from rl.q_table import QTable
from rl.scenario_action_mapper import ScenarioActionMapper
from rl.scenario_suite_coverage_transition import (
    ScenarioSuiteCoverageTransition,
)
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)
from rl.state_encoder import StateEncoder
from rl.training_session import TrainingSession
from rl.training_statistics import TrainingStatistics


SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"
FUNCTION_NAME = "calculate_score"


def create_scenarios() -> tuple[Scenario, ...]:
    """Gerçek eğitim oturumunda kullanılacak senaryoları oluşturur."""
    return (
        Scenario(
            scenario_id="calculate_score_scenario_001",
            name="calculate_score başarılı yürütme yolu",
            path_index=1,
            priority_rank=1,
            priority_level="High",
            dqm_score=100.0,
            node_ids=(1, 3, 4, 2),
            edge_labels=(None, "True", None),
            contains_loop=False,
            contains_exception=False,
            description=(
                "score 85 veya daha büyük olduğunda "
                "Başarılı sonucu beklenir."
            ),
            keyword_arguments=(("score", 85),),
            expected_result="Başarılı",
            expected_exception=None,
        ),
        Scenario(
            scenario_id="calculate_score_scenario_002",
            name="calculate_score orta yürütme yolu",
            path_index=2,
            priority_rank=2,
            priority_level="High",
            dqm_score=100.0,
            node_ids=(1, 3, 5, 6, 2),
            edge_labels=(None, "False", "True", None),
            contains_loop=False,
            contains_exception=False,
            description=(
                "score 50 ile 85 arasında olduğunda "
                "Orta sonucu beklenir."
            ),
            keyword_arguments=(("score", 50),),
            expected_result="Orta",
            expected_exception=None,
        ),
        Scenario(
            scenario_id="calculate_score_scenario_003",
            name="calculate_score başarısız yürütme yolu",
            path_index=3,
            priority_rank=3,
            priority_level="Medium",
            dqm_score=76.47,
            node_ids=(1, 3, 5, 7, 2),
            edge_labels=(None, "False", "False", None),
            contains_loop=False,
            contains_exception=False,
            description=(
                "score 50'nin altında olduğunda "
                "Başarısız sonucu beklenir."
            ),
            keyword_arguments=(("score", 49),),
            expected_result="Başarısız",
            expected_exception=None,
        ),
    )


def create_initial_state() -> CoverageState:
    """Her episode'un başlangıç durumunu oluşturur."""
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4, 5, 6),
        uncovered_branches=4,
    )


def create_training_components(
    tmp_path: Path,
) -> tuple[
    TrainingSession,
    CoverageEnvironment,
    QTable,
    TrainingStatistics,
]:
    """Gerçek eğitim oturumunun bütün bileşenlerini hazırlar."""
    scenarios = create_scenarios()

    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    suite_transition = ScenarioSuiteCoverageTransition(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path,
        overwrite=True,
        timeout_seconds=30.0,
    )

    transition_adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=suite_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=transition_adapter,
        episode_reset_callback=suite_transition.reset,
    )

    q_table = QTable()

    policy = EpsilonGreedyPolicy(
        epsilon=0.0,
        random_generator=random.Random(42),
    )

    agent = QLearningAgent(
        q_table=q_table,
        policy=policy,
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state_encoder = StateEncoder(
        coverage_bucket_size=10.0,
        missing_lines_bucket_size=2,
        uncovered_branches_bucket_size=1,
    )

    trainer = QLearningTrainer(
        agent=agent,
        state_encoder=state_encoder,
    )

    statistics = TrainingStatistics()

    session = TrainingSession(
        trainer=trainer,
        statistics=statistics,
    )

    return (
        session,
        environment,
        q_table,
        statistics,
    )


def test_real_training_session_runs_multiple_episodes(
    tmp_path: Path,
) -> None:
    """
    Gerçek coverage altyapısıyla birden fazla episode çalıştırıldığını
    ve her episode sonucunun istatistiklere kaydedildiğini doğrular.
    """
    (
        session,
        environment,
        q_table,
        statistics,
    ) = create_training_components(tmp_path)

    result = session.run(
        environment=environment,
        episode_count=3,
    )

    assert result.success is True
    assert result.requested_episode_count == 3
    assert result.completed_episode_count == 3
    assert len(result.episodes) == 3

    assert statistics.episode_count == 3

    assert all(
        episode.final_coverage_percentage == 100.0
        for episode in result.episodes
    )

    assert all(
        episode.full_coverage is True
        for episode in result.episodes
    )

    assert all(
        episode.step_count >= 1
        for episode in result.episodes
    )

    assert result.full_coverage_episode_count == 3
    assert result.best_episode is not None

    assert len(q_table) > 0


def test_real_training_session_records_reward_and_step_metrics(
    tmp_path: Path,
) -> None:
    """
    Gerçek episode sonuçlarından reward, coverage ve adım
    istatistiklerinin üretildiğini doğrular.
    """
    (
        session,
        environment,
        _,
        statistics,
    ) = create_training_components(tmp_path)

    result = session.run(
        environment=environment,
        episode_count=3,
    )

    assert statistics.total_reward > 0.0
    assert statistics.average_reward > 0.0
    assert statistics.best_reward > 0.0

    assert statistics.average_step_count > 0.0
    assert statistics.best_step_count > 0

    assert statistics.best_coverage_percentage == 100.0
    assert statistics.full_coverage_episode_count == 3

    best_episode = result.best_episode

    assert best_episode is not None
    assert best_episode.full_coverage is True
    assert best_episode.final_coverage_percentage == 100.0
    assert best_episode.step_count == statistics.best_step_count


def test_real_training_session_preserves_learning_between_runs(
    tmp_path: Path,
) -> None:
    """
    TrainingSession yeniden çalıştırıldığında environment'ın
    sıfırlandığını fakat aynı QTable içerisindeki öğrenmenin
    korunduğunu doğrular.
    """
    (
        session,
        environment,
        q_table,
        statistics,
    ) = create_training_components(tmp_path)

    first_result = session.run(
        environment=environment,
        episode_count=2,
    )

    first_state_count = len(q_table)

    assert first_result.success is True
    assert first_state_count > 0
    assert statistics.episode_count == 2

    second_result = session.run(
        environment=environment,
        episode_count=2,
        clear_statistics=False,
    )

    assert second_result.success is True

    assert len(q_table) >= first_state_count
    assert statistics.episode_count == 4

    assert tuple(
        episode.episode_number
        for episode in statistics.episodes
    ) == (1, 2, 3, 4)

    assert environment.is_done is True
    assert environment.current_state.is_fully_covered is True


def test_real_training_session_creates_cumulative_test_suite(
    tmp_path: Path,
) -> None:
    """
    Eğitim sırasında kümülatif pytest dosyasının oluşturulduğunu
    ve gerçek test çağrılarını içerdiğini doğrular.
    """
    (
        session,
        environment,
        _,
        _,
    ) = create_training_components(tmp_path)

    session.run(
        environment=environment,
        episode_count=1,
    )

    generated_test_file = (
        tmp_path
        / "test_calculate_score_scenario_suite.py"
    )

    assert generated_test_file.exists()

    generated_source = generated_test_file.read_text(
        encoding="utf-8",
    )

    assert "from datasets.sample_code import calculate_score" in (
        generated_source
    )

    assert "result = calculate_score(" in generated_source
    assert "assert result ==" in generated_source
    assert "assert callable" not in generated_source
    assert "TODO" not in generated_source