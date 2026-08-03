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


SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"
FUNCTION_NAME = "calculate_score"


def create_scenarios() -> tuple[Scenario, ...]:
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
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4, 5, 6),
        uncovered_branches=4,
    )


def test_real_multi_episode_training_preserves_q_table(
    tmp_path: Path,
) -> None:
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

    adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=suite_transition,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=adapter,
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

    initial_state_key = state_encoder.encode(
        state=create_initial_state(),
    )

    first_episode_steps = trainer.train_episode(
        environment=environment,
        reset=True,
    )

    assert first_episode_steps
    assert environment.is_done is True
    assert (
        first_episode_steps[-1]
        .state
        .is_fully_covered
        is True
    )

    first_episode_q_values = tuple(
        q_table.get_value(
            state_key=initial_state_key,
            action=action,
        )
        for action in mapper.actions
    )

    assert any(
        value != 0.0
        for value in first_episode_q_values
    )

    second_episode_steps = trainer.train_episode(
        environment=environment,
        reset=True,
    )

    assert second_episode_steps
    assert environment.is_done is True
    assert (
        second_episode_steps[-1]
        .state
        .is_fully_covered
        is True
    )

    second_episode_q_values = tuple(
        q_table.get_value(
            state_key=initial_state_key,
            action=action,
        )
        for action in mapper.actions
    )

    assert any(
        second_value != first_value
        for first_value, second_value in zip(
            first_episode_q_values,
            second_episode_q_values,
            strict=True,
        )
    )

    assert len(q_table) > 0


def test_real_multi_episode_training_resets_environment_but_not_learning(
    tmp_path: Path,
) -> None:
    scenarios = create_scenarios()
    mapper = ScenarioActionMapper(
        scenarios=scenarios,
    )

    suite_transition = ScenarioSuiteCoverageTransition(
        source_file=SOURCE_FILE,
        module_path=MODULE_PATH,
        function_name=FUNCTION_NAME,
        output_directory=tmp_path,
    )

    environment = CoverageEnvironment(
        initial_state=create_initial_state(),
        actions=mapper.actions,
        transition_function=ScenarioTransitionAdapter(
            mapper=mapper,
            transition_function=suite_transition,
        ),
        episode_reset_callback=suite_transition.reset,
    )

    q_table = QTable()

    trainer = QLearningTrainer(
        agent=QLearningAgent(
            q_table=q_table,
            policy=EpsilonGreedyPolicy(
                epsilon=0.0,
                random_generator=random.Random(7),
            ),
            learning_rate=0.5,
            discount_factor=0.9,
        ),
        state_encoder=StateEncoder(
            coverage_bucket_size=10.0,
            missing_lines_bucket_size=2,
            uncovered_branches_bucket_size=1,
        ),
    )

    trainer.train_episode(
        environment=environment,
        reset=True,
    )

    learned_state_count = len(q_table)

    assert learned_state_count > 0

    environment.reset()

    assert (
        environment.current_state.coverage_percentage
        == 0.0
    )
    assert environment.current_state.executed_tests == 0
    assert environment.available_actions == mapper.actions
    assert suite_transition.selected_scenarios == ()

    assert len(q_table) == learned_state_count