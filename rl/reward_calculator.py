from dataclasses import dataclass

from rl.coverage_state import CoverageState


@dataclass(frozen=True, slots=True)
class RewardCalculator:
    """
    Coverage değişimine göre RL ajanının ödülünü hesaplar.

    Ödül yapısı:
    - Coverage artışı pozitif ödül üretir.
    - Coverage değişmemesi küçük bir cezaya neden olur.
    - Coverage düşüşü negatif ödül üretir.
    - Tam coverage elde edilmesi ek bonus sağlar.
    - Her yeni test çalıştırılması küçük bir maliyet oluşturur.
    """

    no_improvement_penalty: float = 1.0
    test_execution_cost: float = 0.1
    full_coverage_bonus: float = 10.0

    def __post_init__(self) -> None:
        self._validate_non_negative(
            "no_improvement_penalty",
            self.no_improvement_penalty,
        )
        self._validate_non_negative(
            "test_execution_cost",
            self.test_execution_cost,
        )
        self._validate_non_negative(
            "full_coverage_bonus",
            self.full_coverage_bonus,
        )

    def calculate(
        self,
        current_state: CoverageState,
        next_state: CoverageState,
    ) -> float:
        coverage_gain = (
            next_state.coverage_percentage
            - current_state.coverage_percentage
        )

        executed_test_difference = max(
            0,
            next_state.executed_tests - current_state.executed_tests,
        )

        reward = coverage_gain

        if coverage_gain == 0:
            reward -= self.no_improvement_penalty

        reward -= executed_test_difference * self.test_execution_cost

        if (
            next_state.is_fully_covered
            and not current_state.is_fully_covered
        ):
            reward += self.full_coverage_bonus

        return reward

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a number.")

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")

        if value < 0:
            raise ValueError(f"{name} cannot be negative.")