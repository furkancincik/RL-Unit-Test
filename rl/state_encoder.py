from dataclasses import dataclass

from rl.coverage_state import CoverageState
from rl.state_key import StateKey


@dataclass(frozen=True, slots=True)
class StateEncoder:
    """
    CoverageState nesnesini Q-Table'da kullanılacak StateKey
    nesnesine dönüştürür.

    Bucket yaklaşımı, birbirine yakın coverage durumlarının aynı
    RL durumu altında değerlendirilmesini sağlar.
    """

    coverage_bucket_size: float = 10.0
    missing_lines_bucket_size: int = 5
    uncovered_branches_bucket_size: int = 5

    def __post_init__(self) -> None:
        self._validate_positive_number(
            name="coverage_bucket_size",
            value=self.coverage_bucket_size,
        )

        self._validate_positive_integer(
            name="missing_lines_bucket_size",
            value=self.missing_lines_bucket_size,
        )

        self._validate_positive_integer(
            name="uncovered_branches_bucket_size",
            value=self.uncovered_branches_bucket_size,
        )

    def encode(
        self,
        state: CoverageState,
    ) -> StateKey:
        """
        CoverageState bilgisini bucket tabanlı StateKey'e dönüştürür.
        """
        if not isinstance(state, CoverageState):
            raise TypeError(
                "state must be a CoverageState instance."
            )

        coverage_bucket = int(
            state.coverage_percentage
            // self.coverage_bucket_size
        )

        missing_lines_bucket = (
            state.missing_line_count
            // self.missing_lines_bucket_size
        )

        uncovered_branches_bucket = (
            state.uncovered_branches
            // self.uncovered_branches_bucket_size
        )

        return StateKey(
            coverage_bucket=coverage_bucket,
            missing_lines_bucket=missing_lines_bucket,
            uncovered_branches_bucket=uncovered_branches_bucket,
        )

    @staticmethod
    def _validate_positive_number(
        name: str,
        value: float,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a number.")

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")

        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    @staticmethod
    def _validate_positive_integer(
        name: str,
        value: int,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")

        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")