from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from analyzer.python_analyzer import FunctionInfo
from cfg.path_analyzer import ExecutionPath
from evaluator.dqm import DQMScore


class JSONReportWriter:
    """Analiz ve DQM sonuÃ§larÄ±nÄ± JSON dosyasÄ±na kaydeder."""

    def write_dqm_report(
        self,
        function: FunctionInfo,
        paths: list[ExecutionPath],
        scores: list[DQMScore],
        output_path: str | Path,
        source_file: str | Path | None = None,
    ) -> Path:
        """
        DQM sonuÃ§larÄ±nÄ± JSON formatÄ±nda kaydeder.

        Args:
            function: Fonksiyon analiz bilgileri.
            paths: Fonksiyona ait yÃ¼rÃ¼tme yollarÄ±.
            scores: YÃ¼rÃ¼tme yollarÄ±nÄ±n DQM skorlarÄ±.
            output_path: OluÅŸturulacak JSON dosyasÄ±nÄ±n yolu.
            source_file: Analiz edilen kaynak dosyanÄ±n yolu.

        Returns:
            OluÅŸturulan JSON dosyasÄ±nÄ±n yolu.
        """
        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_data = {
            "metadata": {
                "source_file": (
                    str(source_file)
                    if source_file is not None
                    else None
                ),
                "generated_at": datetime.now(UTC).isoformat(),
                "report_type": "dqm",
                "schema_version": "1.0",
            },
            "summary": {
                "function_name": function.name,
                "execution_path_count": len(paths),
                "scored_path_count": len(scores),
                "highest_normalized_score": self._highest_score(
                    scores
                ),
                "high_priority_path_count": self._count_priority(
                    scores,
                    "High",
                ),
                "medium_priority_path_count": self._count_priority(
                    scores,
                    "Medium",
                ),
                "low_priority_path_count": self._count_priority(
                    scores,
                    "Low",
                ),
            },
            "function": asdict(function),
            "execution_paths": [
                asdict(execution_path)
                for execution_path in paths
            ],
            "dqm_scores": [
                asdict(score)
                for score in scores
            ],
        }

        path.write_text(
            json.dumps(
                report_data,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

        return path

    @staticmethod
    def _highest_score(scores: list[DQMScore]) -> float:
        """En yÃ¼ksek normalize DQM skorunu dÃ¶ndÃ¼rÃ¼r."""
        if not scores:
            return 0.0

        return max(
            score.normalized_score
            for score in scores
        )

    @staticmethod
    def _count_priority(
        scores: list[DQMScore],
        priority_level: str,
    ) -> int:
        """Belirtilen Ã¶ncelik seviyesindeki yol sayÄ±sÄ±nÄ± dÃ¶ndÃ¼rÃ¼r."""
        return sum(
            score.priority_level == priority_level
            for score in scores
        )
