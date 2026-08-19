from __future__ import annotations

import math
from dataclasses import dataclass, field

from rl.environment_step import EnvironmentStep


@dataclass(frozen=True, slots=True)
class EpisodeStatistics:
    """
    Tek bir RL episode'unun özet sonuçlarını temsil eder.
    """

    episode_number: int
    step_count: int
    total_reward: float
    final_coverage_percentage: float
    full_coverage: bool
    executed_test_count: int
    ordered_action_indices: tuple[int, ...] = ()
    duration_seconds: float | None = None
    done_reason: str | None = None

    def __post_init__(self) -> None:
        self._validate_positive_integer(
            name="episode_number",
            value=self.episode_number,
        )
        self._validate_non_negative_integer(
            name="step_count",
            value=self.step_count,
        )
        if not isinstance(self.ordered_action_indices, tuple):
            raise TypeError("ordered_action_indices tuple olmalıdır.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.ordered_action_indices
        ):
            raise ValueError("ordered_action_indices negatif olmayan tam sayılar içermelidir.")
        if self.duration_seconds is not None:
            self._validate_finite_number("duration_seconds", self.duration_seconds)
            if self.duration_seconds < 0.0:
                raise ValueError("duration_seconds negatif olamaz.")
        if self.done_reason is not None and (
            not isinstance(self.done_reason, str) or not self.done_reason
        ):
            raise TypeError("done_reason string veya None olmalıdır.")
        self._validate_finite_number(
            name="total_reward",
            value=self.total_reward,
        )
        self._validate_percentage(
            name="final_coverage_percentage",
            value=self.final_coverage_percentage,
        )
        self._validate_boolean(
            name="full_coverage",
            value=self.full_coverage,
        )
        self._validate_non_negative_integer(
            name="executed_test_count",
            value=self.executed_test_count,
        )

    @staticmethod
    def _validate_positive_integer(
        name: str,
        value: int,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} bir tam sayı olmalıdır."
            )

        if value < 1:
            raise ValueError(
                f"{name} 1 veya daha büyük olmalıdır."
            )

    @staticmethod
    def _validate_non_negative_integer(
        name: str,
        value: int,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} bir tam sayı olmalıdır."
            )

        if value < 0:
            raise ValueError(
                f"{name} negatif olamaz."
            )

    @staticmethod
    def _validate_finite_number(
        name: str,
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{name} sayısal olmalıdır."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} sonlu bir sayı olmalıdır."
            )

    @staticmethod
    def _validate_percentage(
        name: str,
        value: float,
    ) -> None:
        EpisodeStatistics._validate_finite_number(
            name=name,
            value=value,
        )

        if not 0.0 <= float(value) <= 100.0:
            raise ValueError(
                f"{name} 0 ile 100 arasında olmalıdır."
            )

    @staticmethod
    def _validate_boolean(
        name: str,
        value: bool,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"{name} bool olmalıdır."
            )


@dataclass(slots=True)
class TrainingStatistics:
    """
    Birden fazla RL episode'una ait sonuçları toplar ve
    eğitim sürecine ilişkin özet metrikler üretir.
    """

    _episodes: list[EpisodeStatistics] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def episodes(self) -> tuple[EpisodeStatistics, ...]:
        """Kaydedilmiş episode sonuçlarını döndürür."""
        return tuple(self._episodes)

    @property
    def episode_count(self) -> int:
        """Kaydedilmiş episode sayısını döndürür."""
        return len(self._episodes)

    @property
    def total_reward(self) -> float:
        """Bütün episode reward toplamını döndürür."""
        return round(
            sum(
                episode.total_reward
                for episode in self._episodes
            ),
            4,
        )

    @property
    def average_reward(self) -> float:
        """Episode başına ortalama reward değerini döndürür."""
        if not self._episodes:
            return 0.0

        return round(
            self.total_reward / self.episode_count,
            4,
        )

    @property
    def best_reward(self) -> float:
        """Elde edilen en yüksek episode reward değerini döndürür."""
        if not self._episodes:
            return 0.0

        return max(
            episode.total_reward
            for episode in self._episodes
        )

    @property
    def average_step_count(self) -> float:
        """Episode başına ortalama adım sayısını döndürür."""
        if not self._episodes:
            return 0.0

        return round(
            sum(
                episode.step_count
                for episode in self._episodes
            )
            / self.episode_count,
            2,
        )

    @property
    def best_episode(self) -> EpisodeStatistics | None:
        """
        Proje hedeflerine göre en iyi episode sonucunu döndürür.

        Öncelik sırası:
        1. En yüksek coverage,
        2. Aynı coverage değerinde en az çalıştırılmış test,
        3. Test sayısı eşitse en yüksek reward,
        4. Hâlâ eşitse daha erken episode.
        """
        if not self._episodes:
            return None

        return max(
            self._episodes,
            key=lambda episode: (
                episode.final_coverage_percentage,
                -episode.executed_test_count,
                episode.total_reward,
                -episode.episode_number,
            ),
        )

    @property
    def best_step_count(self) -> int:
        """
        En yüksek coverage değerine ulaşan episode'lar içerisindeki
        en düşük adım sayısını döndürür.

        Hiç episode yoksa 0 döndürülür.
        """
        if not self._episodes:
            return 0

        best_coverage = self.best_coverage_percentage

        best_coverage_episodes = (
            episode
            for episode in self._episodes
            if (
                episode.final_coverage_percentage
                == best_coverage
            )
        )

        return min(
            episode.step_count
            for episode in best_coverage_episodes
        )

    @property
    def best_executed_test_count(self) -> int:
        """
        En yüksek coverage değerine ulaşan episode'lar içerisindeki
        minimum çalıştırılmış test sayısını döndürür.

        Hiç episode yoksa 0 döndürülür.
        """
        if not self._episodes:
            return 0

        best_coverage = self.best_coverage_percentage

        return min(
            episode.executed_test_count
            for episode in self._episodes
            if (
                episode.final_coverage_percentage
                == best_coverage
            )
        )

    @property
    def best_coverage_percentage(self) -> float:
        """
        Episode'lar içinde ulaşılan en yüksek coverage değerini döndürür.
        """
        if not self._episodes:
            return 0.0

        return max(
            episode.final_coverage_percentage
            for episode in self._episodes
        )

    @property
    def full_coverage_episode_count(self) -> int:
        """Tam coverage ile tamamlanan episode sayısını döndürür."""
        return sum(
            episode.full_coverage
            for episode in self._episodes
        )

    def record_episode(
        self,
        steps: tuple[EnvironmentStep, ...],
        *,
        duration_seconds: float | None = None,
    ) -> EpisodeStatistics:
        """
        Tamamlanmış bir episode'un adımlarından istatistik üretir.
        """
        self._validate_steps(steps)

        final_step = steps[-1]

        episode = EpisodeStatistics(
            episode_number=self.episode_count + 1,
            step_count=len(steps),
            total_reward=round(
                sum(
                    step.reward
                    for step in steps
                ),
                4,
            ),
            final_coverage_percentage=(
                final_step.state.coverage_percentage
            ),
            full_coverage=(
                final_step.state.is_fully_covered
            ),
            executed_test_count=(
                final_step.state.executed_tests
            ),
            ordered_action_indices=tuple(
                step.action.scenario_index
                for step in steps
                if step.action is not None
            ),
            duration_seconds=duration_seconds,
            done_reason=final_step.done_reason,
        )

        self._episodes.append(
            episode
        )

        return episode

    def clear(self) -> None:
        """Kaydedilmiş bütün eğitim istatistiklerini temizler."""
        self._episodes.clear()

    @staticmethod
    def _validate_steps(
        steps: tuple[EnvironmentStep, ...],
    ) -> None:
        if not isinstance(steps, tuple):
            raise TypeError(
                "steps bir EnvironmentStep tuple'ı olmalıdır."
            )

        if not steps:
            raise ValueError(
                "Episode en az bir adım içermelidir."
            )

        if any(
            not isinstance(step, EnvironmentStep)
            for step in steps
        ):
            raise TypeError(
                "steps yalnızca EnvironmentStep "
                "nesneleri içermelidir."
            )

        if not steps[-1].done:
            raise ValueError(
                "Episode tamamlanmadan istatistik kaydedilemez."
            )
