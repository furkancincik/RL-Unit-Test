from __future__ import annotations

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
        Tam coverage sağlayan episode'lar arasından en az adımlı
        sonucu döndürür.

        Adım sayıları eşitse reward değeri daha yüksek olan episode
        tercih edilir. Tam coverage sağlayan episode yoksa None döner.
        """
        full_coverage_episodes = [
            episode
            for episode in self.episodes
            if episode.full_coverage
        ]

        if not full_coverage_episodes:
            return None

        return min(
            full_coverage_episodes,
            key=lambda episode: (
                episode.step_count,
                -episode.total_reward,
                episode.episode_number,
            ),
        )


class TrainingSession:
    """
    QLearningTrainer kullanarak birden fazla episode çalıştırır
    ve her episode sonucunu TrainingStatistics içerisinde kaydeder.

    Bu sınıf:
    - QTable oluşturmaz veya sıfırlamaz,
    - Action seçme algoritmasını değiştirmez,
    - CoverageEnvironment geçişlerini yönetmez.

    Aynı trainer ve agent kullanıldığı için QTable episode'lar
    arasında korunur.
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

        Args:
            trainer:
                Her episode'u çalıştıracak QLearningTrainer.

            statistics:
                Episode sonuçlarını saklayacak istatistik bileşeni.
                Verilmezse yeni bir TrainingStatistics oluşturulur.
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
    ) -> TrainingSessionResult:
        """
        Belirtilen sayıda episode çalıştırır.

        Her episode başlamadan önce environment, QLearningTrainer
        tarafından reset edilir. Environment reset edilirken state,
        action listesi ve episode'a özel transition verileri temizlenir.
        QTable ise aynı agent içerisinde korunduğu için öğrenme devam eder.

        Args:
            environment:
                Eğitimin yürütüleceği coverage ortamı.

            episode_count:
                Çalıştırılacak episode sayısı.

            clear_statistics:
                Eğitim başlamadan önce daha önce kaydedilmiş
                istatistiklerin temizlenip temizlenmeyeceği.

        Returns:
            Eğitim oturumunun episode sonuçlarını içeren
            TrainingSessionResult.

        Raises:
            TypeError:
                environment, episode_count veya clear_statistics
                geçersiz türdeyse.

            ValueError:
                episode_count sıfır veya negatifse.
        """
        self._validate_environment(environment)
        self._validate_episode_count(episode_count)
        self._validate_clear_statistics(clear_statistics)

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