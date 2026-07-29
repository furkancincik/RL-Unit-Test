from dataclasses import dataclass

from rl.coverage_state import CoverageState


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    """
    RL ortamında bir aksiyon uygulandıktan sonra oluşan sonucu temsil eder.

    Attributes:
        state:
            Aksiyon sonrasında oluşan yeni coverage durumu.

        reward:
            Aksiyon sonucunda elde edilen ödül değeri.

        done:
            Episode'un tamamlanıp tamamlanmadığını belirtir.
    """

    state: CoverageState
    reward: float
    done: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state, CoverageState):
            raise TypeError("state must be a CoverageState instance.")

        if isinstance(self.reward, bool):
            raise TypeError("reward must be a number.")

        if not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be a number.")

        if not isinstance(self.done, bool):
            raise TypeError("done must be a boolean.")