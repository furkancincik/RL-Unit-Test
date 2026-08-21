from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from models.analysis_job_result import AnalysisJobStatus
from models.external_source_analysis_result import (
    ExternalExecutionPolicy,
    ExternalModuleSelectionMode,
    ExternalSourceKind,
)


TRUSTED_WARNING = (
    "Timeout bir güvenlik sandbox'ı değildir. Dinamik analiz yalnız güvenilen "
    "kaynaklarda kullanılmalıdır."
)


class AnalysisOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: ExternalExecutionPolicy = Field(
        default=ExternalExecutionPolicy.STATIC_DISCOVERY_ONLY,
        description=TRUSTED_WARNING,
    )
    trusted_execution_acknowledged: bool = Field(
        default=False,
        description="TRUSTED_DYNAMIC_ANALYSIS için açık güven onayı.",
    )
    selection_mode: ExternalModuleSelectionMode = ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
    explicit_relative_paths: tuple[str, ...] = ()
    explicit_module_names: tuple[str, ...] = ()
    maximum_module_count: Annotated[int, Field(ge=1, le=100)] = 10
    maximum_function_count: Annotated[int, Field(ge=1, le=100)] = 10
    episode_count: Annotated[int, Field(ge=1, le=1000)] = 3
    random_seed: int | None = 42
    pytest_coverage_timeout_seconds: Annotated[float, Field(gt=0, le=3600)] = 30.0
    function_pipeline_timeout_seconds: Annotated[float | None, Field(gt=0, le=7200)] = 120.0
    greedy_minimization: bool = False
    strategy_comparison: bool = False

    @model_validator(mode="after")
    def validate_contract(self) -> AnalysisOptionsRequest:
        if self.policy is ExternalExecutionPolicy.TRUSTED_DYNAMIC_ANALYSIS:
            if not self.trusted_execution_acknowledged:
                raise ValueError("Trusted dynamic analysis açık acknowledgement gerektirir.")
            if self.function_pipeline_timeout_seconds is None:
                raise ValueError("Trusted dynamic analysis function pipeline timeout gerektirir.")
        if self.selection_mode is ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
            if self.explicit_relative_paths or self.explicit_module_names:
                raise ValueError("All-eligible selection explicit değer kabul etmez.")
        elif self.selection_mode is ExternalModuleSelectionMode.EXPLICIT_RELATIVE_PATHS:
            if not self.explicit_relative_paths or self.explicit_module_names:
                raise ValueError("Explicit relative-path selection yalnız relative path gerektirir.")
        elif not self.explicit_module_names or self.explicit_relative_paths:
            raise ValueError("Explicit module-name selection yalnız module name gerektirir.")
        return self


class InlineJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_code: str = Field(min_length=1, description="Public response'a geri yazılmaz.")
    analysis: AnalysisOptionsRequest = Field(default_factory=AnalysisOptionsRequest)


class GitHubJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository_url: HttpUrl = Field(description="Yalnız mevcut public GitHub acquisition politikası kabul edilir.")
    analysis: AnalysisOptionsRequest = Field(default_factory=AnalysisOptionsRequest)


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    source_kind: ExternalSourceKind
    analysis_policy: ExternalExecutionPolicy
    status: AnalysisJobStatus
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress_stage: str | None
    safe_error_category: str | None
    cancellation_requested: bool
    artifact_count: int


class FunctionResultResponse(BaseModel):
    qualified_name: str
    status: str
    skip_reason: str | None
    scenario_count: int | None
    rl_test_count: int | None
    line_coverage_percent: float | None
    branch_coverage_percent: float | None
    greedy_selected_count: int | None
    rl_selected_count: int | None
    strategy_winner: str | None


class ModuleResultResponse(BaseModel):
    relative_path: str
    module_name: str | None
    status: str
    discovered_function_count: int
    analyzed_function_count: int
    limit_skipped_function_count: int
    functions: list[FunctionResultResponse]


class JobResultResponse(BaseModel):
    job_id: str
    source_kind: ExternalSourceKind
    analysis_policy: ExternalExecutionPolicy
    status: AnalysisJobStatus
    acquisition_status: str | None
    discovered_module_count: int
    selected_module_count: int
    discovered_function_count: int
    analyzed_function_count: int
    limit_skipped_function_count: int
    project_line_coverage_percent: float | None
    project_branch_coverage_percent: float | None
    modules: list[ModuleResultResponse]
    issues: list[str]


class ArtifactResponse(BaseModel):
    artifact_id: str
    filename: str
    content_type: str
    size_bytes: int


class ArtifactListResponse(BaseModel):
    job_id: str
    artifacts: list[ArtifactResponse]


class HealthResponse(BaseModel):
    status: str
    running_jobs: int
    queued_jobs: int
    maximum_active_jobs: int
    authentication_enabled: bool = False
    rate_limiting_enabled: bool = False


class ProblemResponse(BaseModel):
    detail: str
    category: str
