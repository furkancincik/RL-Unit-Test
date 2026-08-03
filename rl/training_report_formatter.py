from __future__ import annotations

from rl.training_session import TrainingSessionResult
from rl.training_statistics import (
    EpisodeStatistics,
    TrainingStatistics,
)


class TrainingReportFormatter:
    """
    RL eğitim oturumu ve episode istatistiklerini okunabilir
    metin raporlarına dönüştürür.

    Bu sınıf eğitim yapmaz, QTable değerlerini değiştirmez ve
    terminale doğrudan çıktı vermez. Yalnızca rapor metni üretir.
    Böylece aynı çıktı CLI, dosya, API veya kullanıcı arayüzünde
    tekrar kullanılabilir.
    """

    def format_episode(
        self,
        episode: EpisodeStatistics,
    ) -> str:
        """
        Tek bir episode sonucunu okunabilir metne dönüştürür.
        """
        self._validate_episode(episode)

        full_coverage_text = (
            "Evet"
            if episode.full_coverage
            else "Hayır"
        )

        return "\n".join(
            (
                f"Episode {episode.episode_number}",
                f"Adım sayısı       : {episode.step_count}",
                f"Toplam reward     : {episode.total_reward:.2f}",
                (
                    "Final coverage   : "
                    f"%{episode.final_coverage_percentage:.2f}"
                ),
                (
                    "Tam coverage     : "
                    f"{full_coverage_text}"
                ),
                (
                    "Çalıştırılan test: "
                    f"{episode.executed_test_count}"
                ),
            )
        )

    def format_session(
        self,
        result: TrainingSessionResult,
        statistics: TrainingStatistics,
    ) -> str:
        """
        Çok episode içeren eğitim oturumunun ayrıntılı raporunu üretir.
        """
        self._validate_result(result)
        self._validate_statistics(statistics)
        self._validate_consistency(
            result=result,
            statistics=statistics,
        )

        sections: list[str] = [
            "RL EĞİTİM OTURUMU",
            "=" * 40,
        ]

        for episode in result.episodes:
            sections.extend(
                (
                    self.format_episode(episode),
                    "-" * 40,
                )
            )

        best_episode = result.best_episode

        if best_episode is None:
            best_episode_text = "Yok"
        else:
            best_episode_text = str(
                best_episode.episode_number
            )

        sections.extend(
            (
                "GENEL ÖZET",
                f"Talep edilen episode : {result.requested_episode_count}",
                f"Tamamlanan episode   : {result.completed_episode_count}",
                (
                    "Tam coverage episode: "
                    f"{result.full_coverage_episode_count}"
                ),
                (
                    "Toplam reward        : "
                    f"{statistics.total_reward:.2f}"
                ),
                (
                    "Ortalama reward      : "
                    f"{statistics.average_reward:.2f}"
                ),
                (
                    "En yüksek reward     : "
                    f"{statistics.best_reward:.2f}"
                ),
                (
                    "Ortalama adım sayısı : "
                    f"{statistics.average_step_count:.2f}"
                ),
                (
                    "En iyi adım sayısı   : "
                    f"{statistics.best_step_count}"
                ),
                (
                    "En yüksek coverage   : "
                    f"%{statistics.best_coverage_percentage:.2f}"
                ),
                (
                    "En iyi episode       : "
                    f"{best_episode_text}"
                ),
                (
                    "Oturum başarılı      : "
                    f"{'Evet' if result.success else 'Hayır'}"
                ),
            )
        )

        return "\n".join(sections)

    @staticmethod
    def _validate_episode(
        episode: EpisodeStatistics,
    ) -> None:
        if not isinstance(
            episode,
            EpisodeStatistics,
        ):
            raise TypeError(
                "episode bir EpisodeStatistics örneği olmalıdır."
            )

    @staticmethod
    def _validate_result(
        result: TrainingSessionResult,
    ) -> None:
        if not isinstance(
            result,
            TrainingSessionResult,
        ):
            raise TypeError(
                "result bir TrainingSessionResult örneği olmalıdır."
            )

    @staticmethod
    def _validate_statistics(
        statistics: TrainingStatistics,
    ) -> None:
        if not isinstance(
            statistics,
            TrainingStatistics,
        ):
            raise TypeError(
                "statistics bir TrainingStatistics örneği olmalıdır."
            )

    @staticmethod
    def _validate_consistency(
        result: TrainingSessionResult,
        statistics: TrainingStatistics,
    ) -> None:
        if result.episodes != statistics.episodes:
            raise ValueError(
                "TrainingSessionResult ile TrainingStatistics "
                "episode kayıtları uyuşmuyor."
            )