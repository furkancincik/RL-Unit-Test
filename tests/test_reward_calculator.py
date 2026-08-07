import pytest

from rl.coverage_state import CoverageState
from rl.reward_calculator import RewardCalculator


def create_state(
    coverage_percentage: float,
    executed_tests: int = 0,
    uncovered_branches: int = 0,
) -> CoverageState:
    """
    RewardCalculator testlerinde kullanılmak üzere CoverageState üretir.
    """
    return CoverageState(
        coverage_percentage=coverage_percentage,
        executed_tests=executed_tests,
        missing_lines=(),
        uncovered_branches=uncovered_branches,
    )


def test_reward_is_positive_when_coverage_increases() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=60.0,
        executed_tests=2,
    )
    next_state = create_state(
        coverage_percentage=72.0,
        executed_tests=3,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(11.9)


def test_reward_is_negative_when_coverage_decreases() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=80.0,
        executed_tests=2,
    )
    next_state = create_state(
        coverage_percentage=75.0,
        executed_tests=3,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(-5.1)


def test_no_improvement_penalty_is_applied() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=65.0,
        executed_tests=2,
    )
    next_state = create_state(
        coverage_percentage=65.0,
        executed_tests=3,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(-1.1)


def test_branch_improvement_adds_reward() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=60.0,
        executed_tests=2,
        uncovered_branches=8,
    )
    next_state = create_state(
        coverage_percentage=70.0,
        executed_tests=3,
        uncovered_branches=5,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(11.4)


def test_branch_only_improvement_is_rewarded_without_no_improvement_penalty() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=70.0,
        executed_tests=2,
        uncovered_branches=6,
    )
    next_state = create_state(
        coverage_percentage=70.0,
        executed_tests=3,
        uncovered_branches=4,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(0.9)


def test_branch_regression_reduces_reward() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=70.0,
        executed_tests=2,
        uncovered_branches=3,
    )
    next_state = create_state(
        coverage_percentage=70.0,
        executed_tests=3,
        uncovered_branches=5,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(-2.1)


def test_higher_branch_improvement_gets_higher_reward_for_same_line_gain() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=60.0,
        executed_tests=2,
        uncovered_branches=8,
    )

    small_branch_gain_state = create_state(
        coverage_percentage=70.0,
        executed_tests=3,
        uncovered_branches=7,
    )

    large_branch_gain_state = create_state(
        coverage_percentage=70.0,
        executed_tests=3,
        uncovered_branches=4,
    )

    small_reward = calculator.calculate(
        current_state=current_state,
        next_state=small_branch_gain_state,
    )

    large_reward = calculator.calculate(
        current_state=current_state,
        next_state=large_branch_gain_state,
    )

    assert large_reward > small_reward


def test_custom_branch_improvement_weight_is_used() -> None:
    calculator = RewardCalculator(
        branch_improvement_weight=2.0,
    )

    current_state = create_state(
        coverage_percentage=50.0,
        executed_tests=1,
        uncovered_branches=5,
    )
    next_state = create_state(
        coverage_percentage=50.0,
        executed_tests=2,
        uncovered_branches=3,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(3.9)


def test_full_coverage_bonus_is_applied() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=95.0,
        executed_tests=4,
    )
    next_state = create_state(
        coverage_percentage=100.0,
        executed_tests=5,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(14.9)


def test_full_coverage_bonus_is_not_applied_twice() -> None:
    calculator = RewardCalculator()

    current_state = create_state(
        coverage_percentage=100.0,
        executed_tests=4,
    )
    next_state = create_state(
        coverage_percentage=100.0,
        executed_tests=5,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(-1.1)


def test_test_execution_cost_is_applied_for_each_new_test() -> None:
    calculator = RewardCalculator(
        test_execution_cost=0.5,
    )

    current_state = create_state(
        coverage_percentage=50.0,
        executed_tests=2,
    )
    next_state = create_state(
        coverage_percentage=60.0,
        executed_tests=4,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(9.0)


def test_no_test_execution_cost_when_test_count_does_not_increase() -> None:
    calculator = RewardCalculator(
        test_execution_cost=0.5,
    )

    current_state = create_state(
        coverage_percentage=50.0,
        executed_tests=3,
    )
    next_state = create_state(
        coverage_percentage=60.0,
        executed_tests=3,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(10.0)


def test_negative_test_difference_does_not_create_bonus() -> None:
    calculator = RewardCalculator(
        test_execution_cost=0.5,
    )

    current_state = create_state(
        coverage_percentage=50.0,
        executed_tests=3,
    )
    next_state = create_state(
        coverage_percentage=60.0,
        executed_tests=2,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(10.0)


def test_custom_reward_parameters_are_used() -> None:
    calculator = RewardCalculator(
        no_improvement_penalty=2.0,
        test_execution_cost=0.5,
        full_coverage_bonus=20.0,
        branch_improvement_weight=1.0,
    )

    current_state = create_state(
        coverage_percentage=90.0,
        executed_tests=2,
        uncovered_branches=2,
    )
    next_state = create_state(
        coverage_percentage=100.0,
        executed_tests=3,
        uncovered_branches=0,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(31.5)


def test_zero_reward_parameters_are_allowed() -> None:
    calculator = RewardCalculator(
        no_improvement_penalty=0.0,
        test_execution_cost=0.0,
        full_coverage_bonus=0.0,
        branch_improvement_weight=0.0,
    )

    current_state = create_state(
        coverage_percentage=50.0,
        executed_tests=1,
        uncovered_branches=5,
    )
    next_state = create_state(
        coverage_percentage=60.0,
        executed_tests=2,
        uncovered_branches=3,
    )

    reward = calculator.calculate(
        current_state=current_state,
        next_state=next_state,
    )

    assert reward == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("no_improvement_penalty", -1.0),
        ("test_execution_cost", -0.1),
        ("full_coverage_bonus", -5.0),
        ("branch_improvement_weight", -0.5),
    ],
)
def test_reward_parameters_cannot_be_negative(
    parameter_name: str,
    parameter_value: float,
) -> None:
    parameters = {
        "no_improvement_penalty": 1.0,
        "test_execution_cost": 0.1,
        "full_coverage_bonus": 10.0,
        "branch_improvement_weight": 0.5,
    }

    parameters[parameter_name] = parameter_value

    with pytest.raises(
        ValueError,
        match=rf"{parameter_name} cannot be negative\.",
    ):
        RewardCalculator(**parameters)


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("no_improvement_penalty", "1"),
        ("test_execution_cost", None),
        ("full_coverage_bonus", True),
        ("no_improvement_penalty", False),
        ("branch_improvement_weight", "0.5"),
        ("branch_improvement_weight", None),
        ("branch_improvement_weight", True),
    ],
)
def test_reward_parameters_must_be_numeric(
    parameter_name: str,
    parameter_value: object,
) -> None:
    parameters = {
        "no_improvement_penalty": 1.0,
        "test_execution_cost": 0.1,
        "full_coverage_bonus": 10.0,
        "branch_improvement_weight": 0.5,
    }

    parameters[parameter_name] = parameter_value

    with pytest.raises(
        TypeError,
        match=rf"{parameter_name} must be a number\.",
    ):
        RewardCalculator(**parameters)  # type: ignore[arg-type]


def test_reward_calculator_is_immutable() -> None:
    calculator = RewardCalculator()

    with pytest.raises(AttributeError):
        calculator.test_execution_cost = 2.0  # type: ignore[misc]
