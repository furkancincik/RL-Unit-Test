from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StateKey:
    """
    Q-Table içerisinde kullanılacak sadeleştirilmiş RL durumunu temsil eder.

    Attributes:
        coverage_bucket:
            Coverage yüzdesinin ait olduğu aralık.

        missing_lines_bucket:
            Eksik satır sayısının ait olduğu aralık.

        uncovered_branches_bucket:
            Kapsanmayan branch sayısının ait olduğu aralık.
    """

    coverage_bucket: int
    missing_lines_bucket: int
    uncovered_branches_bucket: int

    def __post_init__(self) -> None:
        self._validate_bucket(
            name="coverage_bucket",
            value=self.coverage_bucket,
        )
        self._validate_bucket(
            name="missing_lines_bucket",
            value=self.missing_lines_bucket,
        )
        self._validate_bucket(
            name="uncovered_branches_bucket",
            value=self.uncovered_branches_bucket,
        )

    @staticmethod
    def _validate_bucket(
        name: str,
        value: int,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")

        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} cannot be negative.")