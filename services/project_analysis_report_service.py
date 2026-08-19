from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from models.project_analysis_result import ProjectAnalysisResult


class ProjectAnalysisReportWriter:
    """Project sonucunu atomik, makine-okunabilir JSON olarak yazar."""

    def write(self, result: ProjectAnalysisResult) -> Path:
        if not isinstance(result, ProjectAnalysisResult):
            raise TypeError("result bir ProjectAnalysisResult olmalıdır.")
        path = result.report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path


class ProjectAnalysisReportFormatter:
    """Birleşik project sonucunu kısa ve güvenli terminal metnine çevirir."""

    def format(self, result: ProjectAnalysisResult) -> str:
        if not isinstance(result, ProjectAnalysisResult):
            raise TypeError("result bir ProjectAnalysisResult olmalıdır.")
        lines = [
            "=" * 65,
            "PROJE ANALİZ ÖZETİ",
            "=" * 65,
            f"Kaynak dosya          : {result.source_file}",
            f"Seçim modu            : {result.selection_mode.value}",
            f"Proje durumu          : {result.status.value}",
            f"Keşfedilen fonksiyon  : {len(result.discovered_targets)}",
            f"Seçilen fonksiyon     : {result.selected_function_count}",
            f"Çalıştırılan fonksiyon: {result.executed_function_count}",
            (
                "Completed / Partial / Failed / Timed out / Unsupported: "
                f"{result.completed_count} / {result.partial_count} / "
                f"{result.failed_count} / {result.timed_out_count} / "
                f"{result.unsupported_count}"
            ),
            f"Toplam süre           : {result.total_duration_seconds:.3f} saniye",
            "Aggregate coverage    : Ölçülmedi",
            f"JSON raporu           : {result.report_path}",
            "",
            "FONKSİYON SONUÇLARI",
        ]
        for item in result.function_results:
            diagnostic = item.diagnostic
            line_coverage = self._percentage(
                diagnostic.line_coverage_percent
                if diagnostic is not None
                else None
            )
            branch_coverage = self._percentage(
                diagnostic.branch_coverage_percent
                if diagnostic is not None
                else None
            )
            lines.extend(
                (
                    item.target.qualified_name,
                    f"  Durum             : {item.status.value}",
                    f"  Scenario          : {self._value(item.scenario_count)}",
                    f"  Concrete kabul/red: {self._value(item.concrete_accepted_count)} / {self._value(item.concrete_rejected_count)}",
                    f"  RL test / Q-state : {self._value(item.rl_test_count)} / {self._value(item.q_table_state_count)}",
                    f"  Line/branch       : {line_coverage} / {branch_coverage}",
                    (
                        "  Son aşama          : "
                        f"{diagnostic.last_completed_stage.value if diagnostic is not None and diagnostic.last_completed_stage is not None else 'Ölçülmedi'}"
                    ),
                    (
                        "  Durma nedeni       : "
                        f"{item.skip_reason or (diagnostic.error_message if diagnostic is not None else None) or 'Yok'}"
                    ),
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _percentage(value: float | None) -> str:
        return "Ölçülmedi" if value is None else f"%{value:.2f}"

    @staticmethod
    def _value(value: int | None) -> str:
        return "Ölçülmedi" if value is None else str(value)
