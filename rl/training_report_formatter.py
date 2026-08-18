from __future__ import annotations

from models.coverage_reachability_result import (
    FunctionCoverageReachabilityResult,
)
from models.coverage_result import (
    CoverageResult,
    FunctionCoverageResult,
)
from models.pipeline_diagnostic_result import PipelineDiagnosticResult
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

    def format_diagnostic(
        self,
        diagnostic: PipelineDiagnosticResult,
    ) -> str:
        """Tamamlanmamış run dahil güvenli pipeline özetini biçimlendirir."""
        if not isinstance(diagnostic, PipelineDiagnosticResult):
            raise TypeError(
                "diagnostic bir PipelineDiagnosticResult olmalıdır."
            )

        funnel = diagnostic.funnel
        unavailable: list[str] = []
        if diagnostic.line_coverage_percent is None:
            coverage_text = "Ölçülmedi"
            unavailable.append("Line/branch coverage")
        else:
            branch_text = (
                f"%{diagnostic.branch_coverage_percent:.2f}"
                if diagnostic.branch_coverage_percent is not None
                else "Ölçülmedi"
            )
            coverage_text = (
                f"Line %{diagnostic.line_coverage_percent:.2f}, "
                f"branch {branch_text}"
            )

        if funnel.rl_executed_test_count is None:
            unavailable.append("RL çalıştırılan test sayısı")
        if funnel.q_table_state_count is None:
            unavailable.append("Q-Table state sayısı")

        def count_text(value: int | None) -> str:
            return str(value) if value is not None else "Kullanılamıyor"

        sections = (
            "PIPELINE DIAGNOSTIC ÖZETİ",
            "=" * 48,
            f"Run durumu                  : {diagnostic.status.value}",
            (
                "Son tamamlanan aşama         : "
                + (
                    diagnostic.last_completed_stage.value
                    if diagnostic.last_completed_stage is not None
                    else "Yok"
                )
            ),
            (
                "Durma aşaması                : "
                + (
                    diagnostic.stopped_stage.value
                    if diagnostic.stopped_stage is not None
                    else "Yok"
                )
            ),
            f"Durma nedeni                 : {diagnostic.error_message or 'Yok'}",
            "DOĞRULANMIŞ ARA METRİKLER",
            (
                "Bounded path                 : "
                f"{count_text(funnel.bounded_path_count)}"
            ),
            (
                "Pre-concrete scenario        : "
                f"{count_text(funnel.pre_concrete_scenario_count)}"
            ),
            (
                "Concrete kabul / red         : "
                f"{count_text(funnel.concrete_validation_accepted_count)} / "
                f"{count_text(funnel.concrete_validation_rejected_count)}"
            ),
            (
                "Final senaryo havuzu         : "
                f"{count_text(funnel.final_scenario_count)}"
            ),
            f"Coverage ölçümü              : {coverage_text}",
            (
                "Üretilemeyen metrikler       : "
                + (", ".join(unavailable) if unavailable else "Yok")
            ),
        )
        return "\n".join(sections)

    def format_session(
        self,
        result: TrainingSessionResult,
        statistics: TrainingStatistics,
        *,
        function_name: str | None = None,
        coverage_result: (
            CoverageResult | FunctionCoverageResult | None
        ) = None,
        reachability_result: (
            FunctionCoverageReachabilityResult | None
        ) = None,
    ) -> str:
        """
        Çok episode içeren eğitim oturumunun ayrıntılı raporunu üretir.

        ``coverage_result`` verilirse hedef fonksiyon ve dosya geneli
        coverage değerleri ayrı başlıklarla rapora eklenir.
        ``reachability_result`` verilirse doğrulanmış senaryo havuzunun
        coverage sonucu ile bounded path analizi ayrıca raporlanır.
        Parametreler verilmezse eski kullanım korunur.
        """
        self._validate_result(result)
        self._validate_statistics(statistics)
        self._validate_consistency(
            result=result,
            statistics=statistics,
        )
        self._validate_function_name(function_name)
        self._validate_coverage_result(coverage_result)
        self._validate_reachability_result(
            reachability_result=reachability_result,
            coverage_result=coverage_result,
            function_name=function_name,
        )

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

        if reachability_result is not None:
            sections.extend(
                (
                    "-" * 48,
                    "SENARYO HAVUZU VE BOUNDED PATH ÖZETİ",
                    *self._format_reachability_summary(
                        reachability_result
                    ),
                )
            )

        return "\n".join(sections)

    @classmethod
    def _format_reachability_summary(
        cls,
        reachability_result: FunctionCoverageReachabilityResult,
    ) -> tuple[str, ...]:
        """Senaryo havuzu coverage ve bounded path sonucunu biçimlendirir."""
        metadata_text = (
            "Tam"
            if reachability_result.path_metadata_complete
            else "Eksik"
        )
        classification_text = (
            "Evet"
            if reachability_result.classification_complete
            else "Hayır"
        )

        return (
            (
                "Senaryo havuzu satır coverage : "
                f"%{reachability_result.observed_line_coverage_percent:.2f}"
            ),
            (
                "Analiz edilen bounded path    : "
                f"{reachability_result.analyzed_path_count}"
            ),
            (
                "Düğüm ziyaret sınırı          : "
                f"{reachability_result.max_visits_per_node}"
            ),
            (
                "Path metadata durumu          : "
                f"{metadata_text}"
            ),
            cls._format_line_group(
                label="Kapsanan satırlar",
                lines=reachability_result.covered_lines,
            ),
            cls._format_line_group(
                label="FEASIBLE fakat kapsanmayan",
                lines=(
                    reachability_result
                    .feasible_uncovered_lines
                ),
            ),
            cls._format_line_group(
                label="Yalnızca bounded INFEASIBLE",
                lines=reachability_result.infeasible_only_lines,
            ),
            cls._format_line_group(
                label="Çözümlenemeyen satırlar",
                lines=reachability_result.unresolved_lines,
            ),
            (
                "Bounded sınıflandırma tamam mı : "
                f"{classification_text}"
            ),
            (
                "Not: INFEASIBLE_ONLY yalnızca analiz edilen "
                "bounded path kümesi için geçerlidir."
            ),
        )

    @staticmethod
    def _format_line_group(
        *,
        label: str,
        lines: tuple[int, ...],
    ) -> str:
        """Satır grubunu sayı ve satır numaralarıyla biçimlendirir."""
        line_text = (
            ", ".join(str(line) for line in lines)
            if lines
            else "Yok"
        )

        return (
            f"{label:<31}: {len(lines)} "
            f"({line_text})"
        )

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
    def _validate_reachability_result(
        *,
        reachability_result: (
            FunctionCoverageReachabilityResult | None
        ),
        coverage_result: (
            CoverageResult | FunctionCoverageResult | None
        ),
        function_name: str | None,
    ) -> None:
        """Erişilebilirlik özetinin rapor bağlamıyla uyumunu doğrular."""
        if reachability_result is None:
            return

        if not isinstance(
            reachability_result,
            FunctionCoverageReachabilityResult,
        ):
            raise TypeError(
                "reachability_result bir "
                "FunctionCoverageReachabilityResult veya None "
                "olmalıdır."
            )

        reachability_coverage = (
            reachability_result.coverage_result
        )

        if (
            function_name is not None
            and function_name
            != reachability_coverage.function_name
        ):
            raise ValueError(
                "reachability_result farklı bir hedef "
                "fonksiyona aittir."
            )

        if coverage_result is None:
            return

        if not isinstance(
            coverage_result,
            FunctionCoverageResult,
        ):
            raise ValueError(
                "reachability_result yalnızca fonksiyon bazlı "
                "coverage raporuyla birlikte kullanılabilir."
            )

        if (
            coverage_result.source_file
            != reachability_coverage.source_file
            or coverage_result.function_name
            != reachability_coverage.function_name
        ):
            raise ValueError(
                "coverage_result ile reachability_result aynı "
                "kaynak fonksiyona ait olmalıdır."
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
