from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from models.analysis_job_result import AnalysisJobStatus
from models.external_source_analysis_result import (
    ExternalExecutionPolicy,
    ExternalModuleSelectionMode,
    ExternalSourceKind,
)
from models.project_analysis_result import (
    TargetSelectionMode,
    validate_module_identity,
    validate_qualified_target_name,
)


TRUSTED_WARNING = (
    "Timeout bir güvenlik sandbox'ı değildir. Dinamik analiz yalnız güvenilen "
    "kaynaklarda kullanılmalıdır."
)


class QualifiedTargetSelectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module_identity: str
    qualified_name: str

    @field_validator("module_identity")
    @classmethod
    def validate_module(cls, value: str) -> str:
        return validate_module_identity(value)

    @field_validator("qualified_name")
    @classmethod
    def validate_target(cls, value: str) -> str:
        return validate_qualified_target_name(value)


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
    target_selection_mode: TargetSelectionMode = (
        TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT
    )
    explicit_target_names: tuple[str, ...] = ()
    explicit_module_targets: tuple[QualifiedTargetSelectorRequest, ...] = ()
    maximum_module_count: Annotated[int, Field(ge=1, le=100)] = 10
    maximum_function_count: Annotated[int, Field(ge=1, le=100)] = 10
    episode_count: Annotated[int, Field(ge=1, le=1000)] = 3
    random_seed: int | None = 42
    pytest_coverage_timeout_seconds: Annotated[float, Field(gt=0, le=3600)] = 30.0
    function_pipeline_timeout_seconds: Annotated[float | None, Field(gt=0, le=7200)] = 120.0
    project_timeout_seconds: Annotated[float | None, Field(gt=0, le=14_400)] = None
    greedy_minimization: bool = False
    strategy_comparison: bool = False

    @field_validator("explicit_target_names")
    @classmethod
    def validate_target_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_qualified_target_name(value) for value in values)

    @field_validator("project_timeout_seconds", mode="before")
    @classmethod
    def reject_boolean_project_timeout(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Project timeout pozitif ve sonlu sayı olmalıdır.")
        return value

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
        target_explicit = (
            self.target_selection_mode
            is TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS
        )
        has_target_values = bool(
            self.explicit_target_names or self.explicit_module_targets
        )
        if target_explicit != has_target_values:
            raise ValueError(
                "Explicit target selection target değerleriyle kullanılmalıdır."
            )
        if self.explicit_target_names and self.explicit_module_targets:
            raise ValueError(
                "Tek-modül ve çok-modül target seçimleri birlikte kullanılamaz."
            )
        return self


class InlineJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_code: str = Field(min_length=1, description="Public response'a geri yazılmaz.")
    analysis: AnalysisOptionsRequest = Field(default_factory=AnalysisOptionsRequest)

    @field_validator("source_code")
    @classmethod
    def validate_nonblank_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Inline Python source boş bırakılamaz.")
        return value


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
    project_timeout_seconds: float | None
    project_deadline_exceeded: bool
    last_completed_stage: str | None
    deadline_stage: str | None


class InputRejectionCategoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    count: int = Field(ge=0)


class FunctionResultResponse(BaseModel):
    qualified_name: str
    status: str
    skip_reason: str | None
    scenario_count: int | None
    concrete_accepted_count: int | None
    concrete_rejected_count: int | None
    rl_test_count: int | None
    q_table_state_count: int | None
    line_coverage_percent: float | None
    branch_coverage_percent: float | None
    scenario_pool_line_coverage_percent: float | None
    scenario_pool_branch_coverage_percent: float | None
    greedy_line_coverage_percent: float | None
    greedy_branch_coverage_percent: float | None
    greedy_coverage_preserved: bool | None
    best_rl_line_coverage_percent: float | None
    best_rl_branch_coverage_percent: float | None
    best_rl_coverage_preserved: bool | None
    duration_seconds: float | None
    stopped_stage: str | None
    error_category: str | None
    greedy_selected_count: int | None
    rl_selected_count: int | None
    strategy_winner: str | None
    comparison_status: str | None
    comparison_scenario_pool_count: int | None
    greedy_reduction_percentage: float | None
    rl_reduction_percentage: float | None
    coverage_equality_verified: bool | None
    globally_minimal: bool | None
    rl_done_reason: str | None
    bounded_path_count: int = Field(default=0, ge=0)
    input_generation_accepted_count: int = Field(default=0, ge=0)
    input_generation_rejected_count: int = Field(default=0, ge=0)
    input_rejection_categories: list[InputRejectionCategoryResponse] = Field(
        default_factory=list
    )


class ModuleResultResponse(BaseModel):
    relative_path: str
    module_name: str | None
    status: str
    discovered_function_count: int
    analyzed_function_count: int
    limit_skipped_function_count: int
    selection_skipped_function_count: int
    deadline_skipped_function_count: int
    discovered_function_names: list[str]
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
    selection_skipped_function_count: int
    deadline_skipped_function_count: int
    project_line_coverage_percent: float | None
    project_branch_coverage_percent: float | None
    project_coverage: dict[str, Any] | None
    duration_seconds: float
    cleanup_status: str
    modules: list[ModuleResultResponse]
    issues: list[str]
    project_timeout_seconds: float | None
    project_deadline_exceeded: bool
    last_completed_stage: str | None
    deadline_stage: str | None
    completed_function_count: int
    partial_function_count: int
    timed_out_function_count: int


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
