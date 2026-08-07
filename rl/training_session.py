from __future__ import annotations

import math
from dataclasses import dataclass

from rl.coverage_environment import CoverageEnvironment
from rl.q_learning_trainer import QLearningTrainer
from rl.training_statistics import (
    EpisodeStatistics,
    TrainingStatistics,
)


@dataclass(frozen=True, slots=True)
class TrainingSessionResult:
    """
    Çok episode içeren bir RL eğitim oturumunun sonucunu temsil eder.

    Attributes:
        episodes:
            Eğitim sırasında oluşturulan episode istatistikleri.

        requested_episode_count:
            Kullanıcı tarafından talep edilen episode sayısı.

        completed_episode_count:
            Başarıyla tamamlanan episode sayısı.
    """

    episodes: tuple[EpisodeStatistics, ...]
    requested_episode_count: int
    completed_episode_count: int

    @property
    def success(self) -> bool:
        """
        Talep edilen bütün episode'lar tamamlandıysa True döndürür.
        """
        return (
            self.requested_episode_count > 0
            and self.completed_episode_count
            == self.requested_episode_count
        )

    @property
    def full_coverage_episode_count(self) -> int:
        """
        Tam coverage ile tamamlanan episode sayısını döndürür.
        """
        return sum(
            episode.full_coverage
            for episode in self.episodes
        )

    @property
    def best_episode(self) -> EpisodeStatistics | None:
        """
        Proje hedeflerine göre en iyi episode sonucunu döndürür.

        Öncelik sırası:
        1. En yüksek final coverage,
        2. Aynı coverage değerinde en az çalıştırılmış test,
        3. Test sayısı eşitse en yüksek toplam reward,
        4. Hâlâ eşitse daha erken episode.
        """
        if not self.episodes:
            return None

        return max(
            self.episodes,
            key=lambda episode: (
                episode.final_coverage_percentage,
                -episode.executed_test_count,
                episode.total_reward,
                -episode.episode_number,
            ),
        )


class TrainingSession:
    """
    QLearningTrainer kullanarak birden fazla episode çalıştırır
    ve her episode sonucunu TrainingStatistics içerisinde kaydeder.

    Aynı trainer ve agent kullanıldığı için QTable episode'lar
    arasında korunur.

    epsilon_decay_rate verilirse her tamamlanan episode sonrasında
    agent policy'sinin epsilon değeri azaltılır. Böylece eğitim
    başlangıcında daha fazla keşif, ilerleyen episode'larda ise
    daha fazla exploitation yapılabilir.
    """

    __slots__ = (
        "_trainer",
        "_statistics",
    )

    def __init__(
        self,
        trainer: QLearningTrainer,
        statistics: TrainingStatistics | None = None,
    ) -> None:
        """
        Eğitim oturumunun bağımlılıklarını hazırlar.
        """
        self._validate_trainer(trainer)
        self._validate_statistics(statistics)

        self._trainer = trainer
        self._statistics = (
            statistics
            if statistics is not None
            else TrainingStatistics()
        )

    @property
    def trainer(self) -> QLearningTrainer:
        """Eğitimde kullanılan trainer nesnesini döndürür."""
        return self._trainer

    @property
    def statistics(self) -> TrainingStatistics:
        """Eğitim sonuçlarının tutulduğu istatistik nesnesini döndürür."""
        return self._statistics

    def run(
        self,
        environment: CoverageEnvironment,
        episode_count: int,
        *,
        clear_statistics: bool = True,
        epsilon_decay_rate: float | None = None,
        minimum_epsilon: float = 0.0,
    ) -> TrainingSessionResult:
        """
        Belirtilen sayıda episode çalıştırır.

        Args:
            environment:
                Eğitimin yürütüleceği coverage ortamı.

            episode_count:
                Çalıştırılacak episode sayısı.

            clear_statistics:
                Eğitim başlamadan önce eski istatistiklerin
                temizlenip temizlenmeyeceği.

            epsilon_decay_rate:
                Her tamamlanan episode sonrasında epsilon ile
                çarpılacak decay katsayısı. None verilirse epsilon
                değişmez ve eski davranış korunur.

            minimum_epsilon:
                Decay aktifken epsilon değerinin düşebileceği
                minimum keşif oranı.

        Returns:
            Eğitim oturumunun episode sonuçlarını içeren
            TrainingSessionResult.
        """
        self._validate_environment(environment)
        self._validate_episode_count(episode_count)
        self._validate_clear_statistics(clear_statistics)
        self._validate_epsilon_decay_rate(
            epsilon_decay_rate
        )
        self._validate_minimum_epsilon(
            minimum_epsilon
        )

        if clear_statistics:
            self._statistics.clear()

        completed_episodes: list[EpisodeStatistics] = []

        for _ in range(episode_count):
            steps = self._trainer.train_episode(
                environment=environment,
                reset=True,
            )

            episode_statistics = (
                self._statistics.record_episode(
                    steps=steps,
                )
            )

            completed_episodes.append(
                episode_statistics
            )

            if epsilon_decay_rate is not None:
                self._trainer.agent.policy.decay_epsilon(
                    decay_rate=epsilon_decay_rate,
                    minimum_epsilon=minimum_epsilon,
                )

        return TrainingSessionResult(
            episodes=tuple(completed_episodes),
            requested_episode_count=episode_count,
            completed_episode_count=len(
                completed_episodes
            ),
        )

    @staticmethod
    def _validate_trainer(
        trainer: QLearningTrainer,
    ) -> None:
        if not isinstance(
            trainer,
            QLearningTrainer,
        ):
            raise TypeError(
                "trainer bir QLearningTrainer örneği olmalıdır."
            )

    @staticmethod
    def _validate_statistics(
        statistics: TrainingStatistics | None,
    ) -> None:
        if (
            statistics is not None
            and not isinstance(
                statistics,
                TrainingStatistics,
            )
        ):
            raise TypeError(
                "statistics bir TrainingStatistics "
                "örneği veya None olmalıdır."
            )

    @staticmethod
    def _validate_environment(
        environment: CoverageEnvironment,
    ) -> None:
        if not isinstance(
            environment,
            CoverageEnvironment,
        ):
            raise TypeError(
                "environment bir CoverageEnvironment "
                "örneği olmalıdır."
            )

    @staticmethod
    def _validate_episode_count(
        episode_count: int,
    ) -> None:
        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
        ):
            raise TypeError(
                "episode_count bir tam sayı olmalıdır."
            )

        if episode_count < 1:
            raise ValueError(
                "episode_count 1 veya daha büyük olmalıdır."
            )

    @staticmethod
    def _validate_clear_statistics(
        clear_statistics: bool,
    ) -> None:
        if not isinstance(
            clear_statistics,
            bool,
        ):
            raise TypeError(
                "clear_statistics bool olmalıdır."
            )

    @staticmethod
    def _validate_epsilon_decay_rate(
        epsilon_decay_rate: float | None,
    ) -> None:
        if epsilon_decay_rate is None:
            return

        if (
            isinstance(epsilon_decay_rate, bool)
            or not isinstance(
                epsilon_decay_rate,
                (int, float),
            )
        ):
            raise TypeError(
                "epsilon_decay_rate sayısal bir değer "
                "veya None olmalıdır."
            )

        normalized_value = float(
            epsilon_decay_rate
        )

        if not math.isfinite(
            normalized_value
        ):
            raise ValueError(
                "epsilon_decay_rate sonlu olmalıdır."
            )

        if not (
            0.0
            <= normalized_value
            <= 1.0
        ):
            raise ValueError(
                "epsilon_decay_rate 0 ile 1 "
                "arasında olmalıdır."
            )

    @staticmethod
    def _validate_minimum_epsilon(
        minimum_epsilon: float,
    ) -> None:
        if (
            isinstance(minimum_epsilon, bool)
            or not isinstance(
                minimum_epsilon,
                (int, float),
            )
        ):
            raise TypeError(
                "minimum_epsilon sayısal olmalıdır."
            )

        normalized_value = float(
            minimum_epsilon
        )

        if not math.isfinite(
            normalized_value
        ):
            raise ValueError(
                "minimum_epsilon sonlu olmalıdır."
            )

        if not (
            0.0
            <= normalized_value
            <= 1.0
        ):
            raise ValueError(
                "minimum_epsilon 0 ile 1 "
                "arasında olmalıdır."
            )
