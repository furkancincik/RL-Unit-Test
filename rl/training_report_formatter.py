from __future__ import annotations

from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
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

        Episode içerisindeki coverage değeri, RL ortamının hedeflediği
        kapsamı temsil eder. Gerçek RL eğitiminde bu kapsam hedef
        fonksiyondur.
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
                f"Adım sayısı                 : {episode.step_count}",
                f"Toplam reward               : {episode.total_reward:.2f}",
                (
                    "Hedef kapsam coverage       : "
                    f"%{episode.final_coverage_percentage:.2f}"
                ),
                (
                    "Tam hedef kapsam coverage   : "
                    f"{full_coverage_text}"
                ),
                (
                    "Çalıştırılan test           : "
                    f"{episode.executed_test_count}"
                ),
            )
        )

    def format_session(
        self,
        result: TrainingSessionResult,
        statistics: TrainingStatistics,
        *,
        function_name: str | None = None,
        coverage_result: (
            CoverageResult | FunctionCoverageResult | None
        ) = None,
    ) -> str:
        """
        Çok episode içeren eğitim oturumunun ayrıntılı raporunu üretir.

        ``coverage_result`` verilirse hedef fonksiyon ve dosya geneli
        coverage değerleri ayrı başlıklarla rapora eklenir. Parametre
        verilmezse eski kullanım korunur.
        """
        self._validate_result(result)
        self._validate_statistics(statistics)
        self._validate_consistency(
            result=result,
            statistics=statistics,
        )
        self._validate_function_name(function_name)
        self._validate_coverage_result(coverage_result)

        sections: list[str] = [
            "RL EĞİTİM OTURUMU",
            "=" * 48,
        ]

        for episode in result.episodes:
            sections.extend(
                (
                    self.format_episode(episode),
                    "-" * 48,
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
                f"Talep edilen episode         : {result.requested_episode_count}",
                f"Tamamlanan episode           : {result.completed_episode_count}",
                (
                    "Tam hedef coverage episode   : "
                    f"{result.full_coverage_episode_count}"
                ),
                (
                    "Toplam reward                : "
                    f"{statistics.total_reward:.2f}"
                ),
                (
                    "Ortalama reward              : "
                    f"{statistics.average_reward:.2f}"
                ),
                (
                    "En yüksek reward             : "
                    f"{statistics.best_reward:.2f}"
                ),
                (
                    "Ortalama adım sayısı         : "
                    f"{statistics.average_step_count:.2f}"
                ),
                (
                    "En iyi adım sayısı           : "
                    f"{statistics.best_step_count}"
                ),
                (
                    "En yüksek hedef coverage     : "
                    f"%{statistics.best_coverage_percentage:.2f}"
                ),
                (
                    "En iyi episode               : "
                    f"{best_episode_text}"
                ),
                (
                    "Oturum başarılı              : "
                    f"{'Evet' if result.success else 'Hayır'}"
                ),
            )
        )

        if coverage_result is not None:
            sections.extend(
                (
                    "-" * 48,
                    "SON COVERAGE ÖZETİ",
                    *self._format_coverage_summary(
                        coverage_result=coverage_result,
                        function_name=function_name,
                    ),
                )
            )

        return "\n".join(sections)

    @staticmethod
    def _format_coverage_summary(
        *,
        coverage_result: CoverageResult | FunctionCoverageResult,
        function_name: str | None,
    ) -> tuple[str, ...]:
        """Fonksiyon ve dosya coverage sonuçlarını ayrı ayrı biçimlendirir."""
        if isinstance(
            coverage_result,
            FunctionCoverageResult,
        ):
            normalized_function_name = (
                function_name
                or coverage_result.function_name
            )

            function_full_text = (
                "Evet"
                if coverage_result.has_full_coverage
                else "Hayır"
            )
            file_full_text = (
                "Evet"
                if coverage_result.file_coverage.has_full_coverage
                else "Hayır"
            )

            return (
                (
                    f"Hedef fonksiyon              : "
                    f"{normalized_function_name}"
                ),
                (
                    "Fonksiyon satır coverage      : "
                    f"%{coverage_result.line_coverage_percent:.2f}"
                ),
                (
                    "Fonksiyon branch coverage     : "
                    f"%{coverage_result.branch_coverage_percent:.2f}"
                ),
                (
                    "Tam fonksiyon coverage        : "
                    f"{function_full_text}"
                ),
                (
                    "Dosya geneli satır coverage   : "
                    f"%{coverage_result.file_line_coverage_percent:.2f}"
                ),
                (
                    "Dosya geneli branch coverage  : "
                    f"%{coverage_result.file_branch_coverage_percent:.2f}"
                ),
                (
                    "Tam dosya coverage            : "
                    f"{file_full_text}"
                ),
            )

        file_full_text = (
            "Evet"
            if coverage_result.has_full_coverage
            else "Hayır"
        )

        return (
            (
                "Dosya geneli satır coverage   : "
                f"%{coverage_result.line_coverage_percent:.2f}"
            ),
            (
                "Dosya geneli branch coverage  : "
                f"%{coverage_result.branch_coverage_percent:.2f}"
            ),
            (
                "Tam dosya coverage            : "
                f"{file_full_text}"
            ),
        )

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
    def _validate_function_name(
        function_name: str | None,
    ) -> None:
        if function_name is None:
            return

        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string veya None olmalıdır."
            )

        if not function_name.strip():
            raise ValueError(
                "function_name boş olamaz."
            )

    @staticmethod
    def _validate_coverage_result(
        coverage_result: (
            CoverageResult | FunctionCoverageResult | None
        ),
    ) -> None:
        if coverage_result is None:
            return

        if not isinstance(
            coverage_result,
            (
                CoverageResult,
                FunctionCoverageResult,
            ),
        ):
            raise TypeError(
                "coverage_result bir CoverageResult, "
                "FunctionCoverageResult veya None olmalıdır."
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