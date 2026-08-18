from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any


class PipelineRunStatus(str, Enum):
    """Production pipeline çalışmasının tamamlanma durumudur."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class PipelineStage(str, Enum):
    """Production analiz ve RL pipeline aşamalarıdır."""

    SOURCE_ANALYSIS = "SOURCE_ANALYSIS"
    FUNCTION_DISCOVERY = "FUNCTION_DISCOVERY"
    CFG_CONSTRUCTION = "CFG_CONSTRUCTION"
    PATH_DISCOVERY = "PATH_DISCOVERY"
    PATH_FEASIBILITY = "PATH_FEASIBILITY"
    CANDIDATE_GENERATION = "CANDIDATE_GENERATION"
    PATH_INPUT_GENERATION = "PATH_INPUT_GENERATION"
    SCENARIO_GENERATION = "SCENARIO_GENERATION"
    CONCRETE_VALIDATION = "CONCRETE_VALIDATION"
    TEST_GENERATION = "TEST_GENERATION"
    TEST_EXECUTION = "TEST_EXECUTION"
    COVERAGE_MEASUREMENT = "COVERAGE_MEASUREMENT"
    RL_TRAINING = "RL_TRAINING"
    REPORTING = "REPORTING"


@dataclass(frozen=True, slots=True)
class PipelineFunnelSnapshot:
    """Yalnız doğrulanmış pipeline sayaçlarını taşıyan snapshot'tır."""

    bounded_path_count: int | None = None
    feasible_path_count: int | None = None
    infeasible_path_count: int | None = None
    unknown_path_count: int | None = None
    candidate_generated_path_count: int | None = None
    input_generation_accepted_count: int | None = None
    input_generation_rejected_count: int | None = None
    scenario_generation_accepted_count: int | None = None
    scenario_generation_rejected_count: int | None = None
    pre_concrete_scenario_count: int | None = None
    concrete_validation_accepted_count: int | None = None
    concrete_validation_rejected_count: int | None = None
    final_scenario_count: int | None = None
    rl_executed_test_count: int | None = None
    q_table_state_count: int | None = None

    def __post_init__(self) -> None:
        for model_field in fields(self):
            value = getattr(self, model_field.name)
            self._validate_optional_count(model_field.name, value)

        path_counts = (
            self.feasible_path_count,
            self.infeasible_path_count,
            self.unknown_path_count,
        )
        if (
            self.bounded_path_count is not None
            and all(value is not None for value in path_counts)
            and sum(value for value in path_counts if value is not None)
            != self.bounded_path_count
        ):
            raise ValueError(
                "Funnel path sınıflandırma sayıları bounded path "
                "sayısıyla uyuşmalıdır."
            )

        concrete_counts = (
            self.concrete_validation_accepted_count,
            self.concrete_validation_rejected_count,
        )
        if (
            self.pre_concrete_scenario_count is not None
            and all(value is not None for value in concrete_counts)
            and sum(value for value in concrete_counts if value is not None)
            != self.pre_concrete_scenario_count
        ):
            raise ValueError(
                "Funnel concrete validation sayıları pre-concrete "
                "scenario sayısıyla uyuşmalıdır."
            )

        if (
            self.final_scenario_count is not None
            and self.concrete_validation_accepted_count is not None
            and self.final_scenario_count
            != self.concrete_validation_accepted_count
        ):
            raise ValueError(
                "Final scenario sayısı concrete validation kabul "
                "sayısıyla uyuşmalıdır."
            )

    def to_dict(self) -> dict[str, int | None]:
        """Snapshot'ı JSON-compatible sözlüğe dönüştürür."""
        return asdict(self)

    @staticmethod
    def _validate_optional_count(name: str, value: int | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} bir tam sayı veya None olmalıdır.")
        if value < 0:
            raise ValueError(f"{name} negatif olamaz.")


@dataclass(frozen=True, slots=True)
class PipelineDiagnosticResult:
    """Tamamlanmış veya kısmi pipeline çalışmasının güvenli sonucudur."""

    status: PipelineRunStatus
    source_file: Path
    function_name: str
    last_completed_stage: PipelineStage | None
    stopped_stage: PipelineStage | None
    total_duration_seconds: float
    stage_durations: tuple[tuple[PipelineStage, float], ...] = ()
    funnel: PipelineFunnelSnapshot = PipelineFunnelSnapshot()
    error_category: str | None = None
    error_message: str | None = None
    exception_type: str | None = None
    scenario_rejection_counts: tuple[tuple[str, int], ...] = ()
    concrete_rejection_counts: tuple[tuple[str, int], ...] = ()
    line_coverage_percent: float | None = None
    branch_coverage_percent: float | None = None
    reachability_counts: tuple[tuple[str, int], ...] = ()
    includes_sensitive_details: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, PipelineRunStatus):
            raise TypeError("status bir PipelineRunStatus olmalıdır.")
        if not isinstance(self.source_file, Path):
            raise TypeError("source_file bir Path olmalıdır.")
        if not isinstance(self.function_name, str):
            raise TypeError("function_name string olmalıdır.")
        self._validate_optional_stage(
            "last_completed_stage", self.last_completed_stage
        )
        self._validate_optional_stage("stopped_stage", self.stopped_stage)
        self._validate_duration(
            "total_duration_seconds", self.total_duration_seconds
        )
        self._validate_stage_durations(self.stage_durations)
        self._validate_counts(
            "scenario_rejection_counts", self.scenario_rejection_counts
        )
        self._validate_counts(
            "concrete_rejection_counts", self.concrete_rejection_counts
        )
        self._validate_counts("reachability_counts", self.reachability_counts)
        self._validate_optional_percentage(
            "line_coverage_percent", self.line_coverage_percent
        )
        self._validate_optional_percentage(
            "branch_coverage_percent", self.branch_coverage_percent
        )
        if not isinstance(self.includes_sensitive_details, bool):
            raise TypeError("includes_sensitive_details bool olmalıdır.")

    @property
    def success(self) -> bool:
        """Pipeline eksiksiz tamamlandıysa True döndürür."""
        return self.status is PipelineRunStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Sonucu arbitrary nesne taşımayan JSON-compatible yapıya çevirir."""
        return {
            "status": self.status.value,
            "success": self.success,
            "source_file": str(self.source_file),
            "function_name": self.function_name,
            "last_completed_stage": (
                self.last_completed_stage.value
                if self.last_completed_stage is not None
                else None
            ),
            "stopped_stage": (
                self.stopped_stage.value
                if self.stopped_stage is not None
                else None
            ),
            "error_category": self.error_category,
            "error_message": self.error_message,
            "exception_type": self.exception_type,
            "total_duration_seconds": self.total_duration_seconds,
            "stage_durations": {
                stage.value: duration for stage, duration in self.stage_durations
            },
            "funnel": self.funnel.to_dict(),
            "scenario_rejection_counts": dict(
                self.scenario_rejection_counts
            ),
            "concrete_rejection_counts": dict(
                self.concrete_rejection_counts
            ),
            "line_coverage_percent": self.line_coverage_percent,
            "branch_coverage_percent": self.branch_coverage_percent,
            "reachability_counts": dict(self.reachability_counts),
            "includes_sensitive_details": self.includes_sensitive_details,
        }

    @classmethod
    def timed_out(
        cls,
        *,
        source_file: Path,
        function_name: str,
        stopped_stage: PipelineStage,
        last_completed_stage: PipelineStage | None,
        total_duration_seconds: float,
        funnel: PipelineFunnelSnapshot = PipelineFunnelSnapshot(),
        message: str = "Pipeline zaman aşımına uğradı.",
        stage_durations: tuple[tuple[PipelineStage, float], ...] = (),
    ) -> PipelineDiagnosticResult:
        """Gelecekteki timeout orchestration için partial sonuç üretir."""
        return cls(
            status=PipelineRunStatus.TIMED_OUT,
            source_file=source_file,
            function_name=function_name,
            last_completed_stage=last_completed_stage,
            stopped_stage=stopped_stage,
            error_category="TIMEOUT",
            error_message=message,
            exception_type=None,
            total_duration_seconds=total_duration_seconds,
            stage_durations=stage_durations,
            funnel=funnel,
        )

    @staticmethod
    def _validate_optional_stage(
        name: str, value: PipelineStage | None
    ) -> None:
        if value is not None and not isinstance(value, PipelineStage):
            raise TypeError(f"{name} bir PipelineStage veya None olmalıdır.")

    @staticmethod
    def _validate_duration(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} sayısal olmalıdır.")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} negatif olmayan sonlu sayı olmalıdır.")

    @classmethod
    def _validate_stage_durations(
        cls, values: tuple[tuple[PipelineStage, float], ...]
    ) -> None:
        if not isinstance(values, tuple):
            raise TypeError("stage_durations tuple olmalıdır.")
        seen: set[PipelineStage] = set()
        for stage, duration in values:
            if not isinstance(stage, PipelineStage):
                raise TypeError("Aşama süresi anahtarı PipelineStage olmalıdır.")
            if stage in seen:
                raise ValueError("Aşama süreleri yinelenen aşama içeremez.")
            seen.add(stage)
            cls._validate_duration(stage.value, duration)

    @staticmethod
    def _validate_counts(
        name: str, values: tuple[tuple[str, int], ...]
    ) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} tuple olmalıdır.")
        keys: set[str] = set()
        for key, count in values:
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} boş olmayan string anahtar ister.")
            if key in keys:
                raise ValueError(f"{name} yinelenen kategori içeremez.")
            keys.add(key)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} sayaçları tam sayı olmalıdır.")
            if count < 0:
                raise ValueError(f"{name} sayaçları negatif olamaz.")

    @staticmethod
    def _validate_optional_percentage(
        name: str, value: float | None
    ) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} sayısal veya None olmalıdır.")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 100.0:
            raise ValueError(f"{name} 0 ile 100 arasında olmalıdır.")
