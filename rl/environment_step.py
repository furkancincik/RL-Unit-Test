from __future__ import annotations

import math
from dataclasses import dataclass

from rl.coverage_state import CoverageState


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    """
    RL ortamında bir aksiyon uygulandıktan sonra oluşan
    sonucu temsil eder.

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
        """Environment step alanlarını doğrular."""
        self._validate_state(
            self.state
        )
        self._validate_reward(
            self.reward
        )
        self._validate_done(
            self.done
        )

    @staticmethod
    def _validate_state(
        state: CoverageState,
    ) -> None:
        """Coverage state değerini doğrular."""
        if not isinstance(
            state,
            CoverageState,
        ):
            raise TypeError(
                "state must be a CoverageState instance."
            )

    @staticmethod
    def _validate_reward(
        reward: float,
    ) -> None:
        """Reward değerinin geçerli ve sonlu olduğunu doğrular."""
        if (
            isinstance(reward, bool)
            or not isinstance(
                reward,
                (int, float),
            )
        ):
            raise TypeError(
                "reward must be a number."
            )

        if not math.isfinite(
            float(reward)
        ):
            raise ValueError(
                "reward must be finite."
            )

    @staticmethod
    def _validate_done(
        done: bool,
    ) -> None:
        """Episode tamamlanma bilgisini doğrular."""
        if not isinstance(
            done,
            bool,
        ):
            raise TypeError(
                "done must be a boolean."
            )