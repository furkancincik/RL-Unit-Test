import random

from rl.action import Action
from rl.coverage_state import CoverageState
from rl.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl.q_table import QTable
from rl.state_encoder import StateEncoder
from rl.state_key import StateKey


class ControlledRandom(random.Random):
    """
    Entegrasyon testinde seçim davranışını kontrol etmek için kullanılır.
    """

    def __init__(
        self,
        random_value: float,
        choice_index: int = 0,
    ) -> None:
        super().__init__()

        self._random_value = random_value
        self._choice_index = choice_index

    def random(self) -> float:
        return self._random_value

    def choice(self, sequence):  # type: ignore[no-untyped-def]
        return sequence[self._choice_index]


def test_rl_pipeline_encodes_coverage_state() -> None:
    """
    CoverageState bilgisinin StateEncoder tarafından
    doğru StateKey değerine dönüştürüldüğünü doğrular.
    """
    coverage_state = CoverageState(
        coverage_percentage=82.0,
        executed_tests=4,
        missing_lines=(
            10,
            12,
            15,
            18,
            22,
            25,
            30,
            35,
        ),
        uncovered_branches=7,
    )

    encoder = StateEncoder()

    state_key = encoder.encode(
        state=coverage_state,
    )

    assert state_key == StateKey(
        coverage_bucket=8,
        missing_lines_bucket=1,
        uncovered_branches_bucket=1,
    )


def test_rl_pipeline_selects_best_action() -> None:
    """
    CoverageState → StateEncoder → QTable →
    EpsilonGreedyPolicy akışını birlikte doğrular.
    """
    coverage_state = CoverageState(
        coverage_percentage=82.0,
        executed_tests=4,
        missing_lines=(
            10,
            12,
            15,
            18,
            22,
            25,
            30,
            35,
        ),
        uncovered_branches=7,
    )

    encoder = StateEncoder()

    state_key = encoder.encode(
        state=coverage_state,
    )

    first_action = Action(
        scenario_index=0,
    )

    second_action = Action(
        scenario_index=1,
    )

    third_action = Action(
        scenario_index=2,
    )

    actions = (
        first_action,
        second_action,
        third_action,
    )

    q_table = QTable()

    q_table.set_value(
        state_key=state_key,
        action=first_action,
        value=2.0,
    )

    q_table.set_value(
        state_key=state_key,
        action=second_action,
        value=9.0,
    )

    q_table.set_value(
        state_key=state_key,
        action=third_action,
        value=4.0,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=0.0,
        random_generator=ControlledRandom(
            random_value=0.50,
        ),
    )

    selected_action = policy.select_action(
        state_key=state_key,
        actions=actions,
        q_table=q_table,
    )

    assert selected_action == second_action


def test_rl_pipeline_can_explore_random_action() -> None:
    """
    Epsilon değeri 1.0 olduğunda politikanın Q değerlerinden
    bağımsız olarak rastgele aksiyon seçebildiğini doğrular.
    """
    coverage_state = CoverageState(
        coverage_percentage=60.0,
        executed_tests=2,
        missing_lines=(
            5,
            8,
            11,
            14,
            20,
        ),
        uncovered_branches=3,
    )

    state_key = StateEncoder().encode(
        state=coverage_state,
    )

    first_action = Action(
        scenario_index=0,
    )

    second_action = Action(
        scenario_index=1,
    )

    actions = (
        first_action,
        second_action,
    )

    q_table = QTable()

    q_table.set_value(
        state_key=state_key,
        action=first_action,
        value=100.0,
    )

    q_table.set_value(
        state_key=state_key,
        action=second_action,
        value=0.0,
    )

    policy = EpsilonGreedyPolicy(
        epsilon=1.0,
        random_generator=ControlledRandom(
            random_value=0.25,
            choice_index=1,
        ),
    )

    selected_action = policy.select_action(
        state_key=state_key,
        actions=actions,
        q_table=q_table,
    )

    assert selected_action == second_action