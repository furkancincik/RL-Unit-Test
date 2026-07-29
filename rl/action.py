from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    """
    Reinforcement Learning ajanının seçebileceği bir test senaryosunu temsil eder.

    Attributes:
        scenario_index:
            Çalıştırılacak test senaryosunun sıfır tabanlı indeksidir.
    """

    scenario_index: int

    def __post_init__(self) -> None:
        if isinstance(self.scenario_index, bool):
            raise TypeError("scenario_index must be an integer.")

        if not isinstance(self.scenario_index, int):
            raise TypeError("scenario_index must be an integer.")

        if self.scenario_index < 0:
            raise ValueError("scenario_index cannot be negative.")