from __future__ import annotations

from dataclasses import dataclass

from analyzer.python_analyzer import FunctionInfo
from cfg.path_analyzer import ExecutionPath


@dataclass
class DQMScore:
    """Bir yÃ¼rÃ¼tme yoluna ait DQM deÄŸerlendirme sonucunu temsil eder."""

    path_index: int
    path_length: int
    decision_edge_count: int
    contains_loop: bool
    contains_exception: bool
    raw_score: float
    normalized_score: float
    priority_level: str


class DecisionQualityMatrix:
    """YÃ¼rÃ¼tme yollarÄ±nÄ± test Ã¶nceliÄŸine gÃ¶re puanlar."""

    DECISION_LABELS = {
        "True",
        "False",
        "Iterate",
        "Complete",
        "Success",
        "Exception",
    }

    LOOP_LABELS = {
        "Loop",
        "Next",
        "Iterate",
    }

    EXCEPTION_LABELS = {
        "Exception",
    }

    RISK_WEIGHTS = {
        "Low": 1.0,
        "Medium": 1.5,
        "High": 2.0,
    }

    def evaluate_paths(
        self,
        function: FunctionInfo,
        paths: list[ExecutionPath],
    ) -> list[DQMScore]:
        """
        Fonksiyona ait yÃ¼rÃ¼tme yollarÄ±nÄ± DQM yaklaÅŸÄ±mÄ±yla puanlar.

        Args:
            function: Fonksiyonun statik analiz bilgileri.
            paths: CFG Ã¼zerinden Ã§Ä±karÄ±lan yÃ¼rÃ¼tme yollarÄ±.

        Returns:
            Her yol iÃ§in oluÅŸturulan DQM skorlarÄ±nÄ±n listesi.
        """
        if not paths:
            return []

        raw_scores = [
            self._calculate_raw_score(function, path)
            for path in paths
        ]

        maximum_score = max(raw_scores)

        results: list[DQMScore] = []

        for index, (path, raw_score) in enumerate(
            zip(paths, raw_scores),
            start=1,
        ):
            normalized_score = self._normalize_score(
                raw_score=raw_score,
                maximum_score=maximum_score,
            )

            results.append(
                DQMScore(
                    path_index=index,
                    path_length=len(path.node_ids),
                    decision_edge_count=self._count_decision_edges(path),
                    contains_loop=self._contains_loop(path),
                    contains_exception=self._contains_exception(path),
                    raw_score=round(raw_score, 2),
                    normalized_score=round(normalized_score, 2),
                    priority_level=self._determine_priority_level(
                        normalized_score
                    ),
                )
            )

        return sorted(
            results,
            key=lambda result: result.normalized_score,
            reverse=True,
        )

    def _calculate_raw_score(
        self,
        function: FunctionInfo,
        path: ExecutionPath,
    ) -> float:
        """Tek bir yÃ¼rÃ¼tme yolu iÃ§in ham DQM skoru hesaplar."""
        path_length_score = len(path.node_ids) * 1.0

        decision_score = (
            self._count_decision_edges(path) * 3.0
        )

        loop_score = 4.0 if self._contains_loop(path) else 0.0

        exception_score = (
            5.0 if self._contains_exception(path) else 0.0
        )

        complexity_score = (
            function.cyclomatic_complexity * 2.0
        )

        risk_weight = self.RISK_WEIGHTS.get(
            function.risk_level,
            1.0,
        )

        base_score = (
            path_length_score
            + decision_score
            + loop_score
            + exception_score
            + complexity_score
        )

        return base_score * risk_weight

    @staticmethod
    def _count_decision_edges(path: ExecutionPath) -> int:
        """Karar bildiren kenar etiketlerinin sayÄ±sÄ±nÄ± hesaplar."""
        return sum(
            label
            in DecisionQualityMatrix.DECISION_LABELS
            for label in path.edge_labels
        )

    @staticmethod
    def _contains_loop(path: ExecutionPath) -> bool:
        """Yolun dÃ¶ngÃ¼ kenarÄ± iÃ§erip iÃ§ermediÄŸini kontrol eder."""
        return any(
            label in DecisionQualityMatrix.LOOP_LABELS
            for label in path.edge_labels
        )

    @staticmethod
    def _contains_exception(path: ExecutionPath) -> bool:
        """Yolun istisna kenarÄ± iÃ§erip iÃ§ermediÄŸini kontrol eder."""
        return any(
            label in DecisionQualityMatrix.EXCEPTION_LABELS
            for label in path.edge_labels
        )

    @staticmethod
    def _normalize_score(
        raw_score: float,
        maximum_score: float,
    ) -> float:
        """Ham skoru 0â€“100 aralÄ±ÄŸÄ±na normalize eder."""
        if maximum_score <= 0:
            return 0.0

        return (raw_score / maximum_score) * 100.0

    @staticmethod
    def _determine_priority_level(
        normalized_score: float,
    ) -> str:
        """Normalize edilmiÅŸ skora gÃ¶re Ã¶ncelik seviyesini belirler."""
        if normalized_score >= 80:
            return "High"

        if normalized_score >= 50:
            return "Medium"

        return "Low"

