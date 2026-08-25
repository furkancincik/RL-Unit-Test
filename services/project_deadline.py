from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectDeadline:
    """Bir external analysis run'ına ait monotonic toplam süre bütçesi."""

    timeout_seconds: float | None
    started_at: float
    clock: Callable[[], float]

    def __post_init__(self) -> None:
        timeout = self.timeout_seconds
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("project_timeout_seconds sayısal olmalıdır.")
            if not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
                raise ValueError(
                    "project_timeout_seconds pozitif ve sonlu olmalıdır."
                )
            object.__setattr__(self, "timeout_seconds", float(timeout))
        if isinstance(self.started_at, bool) or not isinstance(
            self.started_at, (int, float)
        ):
            raise TypeError("started_at sayısal olmalıdır.")
        if not math.isfinite(float(self.started_at)):
            raise ValueError("started_at sonlu olmalıdır.")
        if not callable(self.clock):
            raise TypeError("clock callable olmalıdır.")

    @classmethod
    def start(
        cls,
        timeout_seconds: float | None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> ProjectDeadline:
        return cls(
            timeout_seconds=timeout_seconds,
            started_at=float(clock()),
            clock=clock,
        )

    def elapsed_seconds(self) -> float:
        return max(0.0, float(self.clock()) - float(self.started_at))

    def remaining_seconds(self) -> float | None:
        if self.timeout_seconds is None:
            return None
        return max(0.0, self.timeout_seconds - self.elapsed_seconds())

    def exceeded(self) -> bool:
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= 0.0

    def clamp_timeout(self, configured_timeout: float | None) -> float | None:
        if configured_timeout is not None:
            if isinstance(configured_timeout, bool) or not isinstance(
                configured_timeout, (int, float)
            ):
                raise TypeError("stage timeout sayısal olmalıdır.")
            if not math.isfinite(float(configured_timeout)) or configured_timeout <= 0:
                raise ValueError("stage timeout pozitif ve sonlu olmalıdır.")
            configured_timeout = float(configured_timeout)
        remaining = self.remaining_seconds()
        if remaining is None:
            return configured_timeout
        if configured_timeout is None:
            return remaining
        return min(configured_timeout, remaining)
