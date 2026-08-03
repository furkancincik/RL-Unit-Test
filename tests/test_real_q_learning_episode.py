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
from rl.scenario_coverage_transition import (
    ScenarioCoverageTransition,
)
from rl.scenario_transition_adapter import (
    ScenarioTransitionAdapter,
)
from rl.state_encoder import StateEncoder


SOURCE_FILE = Path("datasets/sample_code.py")
MODULE_PATH = "datasets.sample_code"
FUNCTION_NAME = "calculate_score"


def create_scenario() -> Scenario:
    """
    Gerçek RL episode testinde kullanılacak test senaryosunu oluşturur.
    """
    return Scenario(
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
            "score değeri 85 veya daha büyük olduğunda "
            "Başarılı sonucu beklenir."
        ),
        keyword_arguments=(
            ("score", 85),
        ),
        expected_result="Başarılı",
        expected_exception=None,
    )


def create_initial_state() -> CoverageState:
    """
    Gerçek coverage ölçümü öncesindeki başlangıç RL durumunu oluşturur.
    """
    return CoverageState(
        coverage_percentage=0.0,
        executed_tests=0,
        missing_lines=(1, 2, 3, 4, 5, 6),
        uncovered_branches=4,
    )


def test_real_q_learning_episode_uses_actual_coverage(
    tmp_path: Path,
) -> None:
    """
    RL ajanının seçtiği aksiyonun gerçek senaryoya dönüştürüldüğünü,
    pytest ve coverage hattında çalıştırıldığını ve elde edilen
    reward ile QTable değerinin güncellendiğini doğrular.
    """
    scenario = create_scenario()

    mapper = ScenarioActionMapper(
        scenarios=(scenario,),
    )

    scenario_coverage_transition = (
        ScenarioCoverageTransition(
            source_file=SOURCE_FILE,
            module_path=MODULE_PATH,
            function_name=FUNCTION_NAME,
            output_directory=tmp_path,
            overwrite=True,
            timeout_seconds=30.0,
        )
    )

    transition_adapter = ScenarioTransitionAdapter(
        mapper=mapper,
        transition_function=scenario_coverage_transition,
    )

    initial_state = create_initial_state()

    environment = CoverageEnvironment(
        initial_state=initial_state,
        actions=mapper.actions,
        transition_function=transition_adapter,
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
        state=initial_state,
    )

    selected_action = mapper.actions[0]

    old_q_value = q_table.get_value(
        state_key=initial_state_key,
        action=selected_action,
    )

    steps = trainer.train_episode(
        environment=environment,
        reset=True,
    )

    new_q_value = q_table.get_value(
        state_key=initial_state_key,
        action=selected_action,
    )

    assert len(steps) == 1

    step = steps[0]

    assert old_q_value == 0.0
    assert new_q_value > old_q_value

    assert step.state.executed_tests == 1
    assert step.state.coverage_percentage > 0.0
    assert step.reward > 0.0
    assert step.done is True

    assert environment.current_state is step.state
    assert environment.available_actions == ()
    assert environment.is_done is True

    generated_test_file = (
        tmp_path
        / "test_calculate_score_scenario_001.py"
    )

    assert generated_test_file.exists()

    generated_source = generated_test_file.read_text(
        encoding="utf-8",
    )

    assert (
        "result = calculate_score(score=85)"
        in generated_source
    )
    assert (
        "assert result == 'Başarılı'"
        in generated_source
    )
    assert "assert callable" not in generated_source
    assert "TODO" not in generated_source