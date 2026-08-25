from __future__ import annotations

import math

import pytest

from services.project_deadline import ProjectDeadline


class FakeMonotonicClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_unlimited_deadline_preserves_configured_stage_timeout() -> None:
    clock = FakeMonotonicClock(12.0)
    deadline = ProjectDeadline.start(None, clock=clock)

    clock.value = 1_000.0

    assert deadline.timeout_seconds is None
    assert deadline.remaining_seconds() is None
    assert deadline.exceeded() is False
    assert deadline.clamp_timeout(30.0) == pytest.approx(30.0)
    assert deadline.clamp_timeout(None) is None


@pytest.mark.parametrize(
    "value,expected_exception",
    (
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
        ("10", TypeError),
    ),
)
def test_project_deadline_rejects_invalid_timeout(
    value: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        ProjectDeadline.start(value, clock=FakeMonotonicClock())  # type: ignore[arg-type]


def test_remaining_time_uses_monotonic_clock_and_clamps_stage_timeout() -> None:
    clock = FakeMonotonicClock(100.0)
    deadline = ProjectDeadline.start(8.0, clock=clock)

    clock.value = 103.5

    assert deadline.remaining_seconds() == pytest.approx(4.5)
    assert deadline.clamp_timeout(20.0) == pytest.approx(4.5)
    assert deadline.clamp_timeout(2.0) == pytest.approx(2.0)
    assert deadline.clamp_timeout(None) == pytest.approx(4.5)

    clock.value = 108.0
    assert deadline.remaining_seconds() == 0.0
    assert deadline.exceeded() is True
    assert deadline.clamp_timeout(2.0) == 0.0


def test_each_deadline_instance_has_isolated_start_state() -> None:
    clock = FakeMonotonicClock(4.0)
    first = ProjectDeadline.start(2.0, clock=clock)
    clock.value = 10.0
    second = ProjectDeadline.start(20.0, clock=clock)

    assert first.exceeded() is True
    assert second.remaining_seconds() == pytest.approx(20.0)
