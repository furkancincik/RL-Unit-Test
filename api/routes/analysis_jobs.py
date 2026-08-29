from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import ValidationError

from analyzer.python_source_reader import (
    PythonSourceEncodingError,
    decode_python_source_bytes,
)

from api.schemas.analysis_jobs import (
    AnalysisOptionsRequest,
    ArtifactListResponse,
    ArtifactResponse,
    GitHubJobRequest,
    HealthResponse,
    InlineJobRequest,
    JobResultResponse,
    JobStatusResponse,
)
from models.external_source_analysis_result import (
    ExternalAnalysisConfiguration,
    ExternalModuleSelection,
    ExternalModuleSelectionMode,
    ExternalSourcePolicyValidationError,
    ExternalSourceAnalysisRequest,
    InlinePythonSource,
    PublicGitHubRepository,
    UploadedPythonFile,
)
from models.project_analysis_result import (
    QualifiedTargetSelector,
    TargetSelection,
    TargetSelectionMode,
)
from services.analysis_job_service import (
    AnalysisArtifactNotFoundError,
    AnalysisJobNotFoundError,
    AnalysisJobQueueFullError,
    AnalysisJobService,
    AnalysisJobStateConflictError,
)
from services.external_source_analysis_service import (
    portable_upload_module_identity,
)
from services.source_acquisition_service import (
    SourceAcquisitionService,
    SourceAcquisitionValidationError,
)


router = APIRouter(prefix="/api/v1")


def _service(request: Request) -> AnalysisJobService:
    return request.app.state.analysis_job_service


def _target_selection(
    options: AnalysisOptionsRequest,
    source: object,
) -> TargetSelection:
    if options.target_selection_mode is TargetSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
        return TargetSelection()
    if isinstance(source, PublicGitHubRepository):
        if options.explicit_target_names or not options.explicit_module_targets:
            raise HTTPException(
                status_code=422,
                detail="GitHub target seçimi module identity gerektirir.",
            )
        selectors = tuple(
            QualifiedTargetSelector(item.module_identity, item.qualified_name)
            for item in options.explicit_module_targets
        )
    else:
        if options.explicit_module_targets or not options.explicit_target_names:
            raise HTTPException(
                status_code=422,
                detail="Inline/upload target seçimi yalnız qualified target adı kabul eder.",
            )
        if isinstance(source, InlinePythonSource):
            module_identity = "inline_source"
        elif isinstance(source, UploadedPythonFile):
            module_identity = portable_upload_module_identity(source)
        else:
            raise HTTPException(
                status_code=422,
                detail="Target selection kaynak türü için desteklenmiyor.",
            )
        selectors = tuple(
            QualifiedTargetSelector(module_identity, qualified_name)
            for qualified_name in options.explicit_target_names
        )
    return TargetSelection(
        TargetSelectionMode.EXPLICIT_QUALIFIED_TARGETS,
        selectors,
    )


def _configuration(
    options: AnalysisOptionsRequest,
    output_root: Path,
    source: object,
) -> ExternalAnalysisConfiguration:
    if options.selection_mode is ExternalModuleSelectionMode.ALL_ELIGIBLE_WITH_LIMIT:
        values: tuple[str, ...] = ()
    elif options.selection_mode is ExternalModuleSelectionMode.EXPLICIT_RELATIVE_PATHS:
        values = tuple(dict.fromkeys(options.explicit_relative_paths))
    else:
        values = tuple(dict.fromkeys(options.explicit_module_names))
    return ExternalAnalysisConfiguration(
        output_root=output_root,
        module_selection=ExternalModuleSelection(options.selection_mode, values),
        target_selection=_target_selection(options, source),
        maximum_selected_modules=options.maximum_module_count,
        maximum_functions_per_module=options.maximum_function_count,
        episode_count=options.episode_count,
        random_seed=options.random_seed,
        pytest_coverage_timeout_seconds=options.pytest_coverage_timeout_seconds,
        per_function_pipeline_timeout_seconds=options.function_pipeline_timeout_seconds,
        project_timeout_seconds=options.project_timeout_seconds,
        run_greedy_baseline=options.greedy_minimization or options.strategy_comparison,
        run_strategy_comparison=options.strategy_comparison,
    )


def _submit(service: AnalysisJobService, source: object, options: AnalysisOptionsRequest) -> JobStatusResponse:
    try:
        request = ExternalSourceAnalysisRequest(
            source=source,
            execution_policy=options.policy,
            configuration=_configuration(
                options, service.settings.output_root, source
            ),
        )
    except ExternalSourcePolicyValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        summary = service.submit(request)
    except AnalysisJobQueueFullError as error:
        raise HTTPException(status_code=429, detail="Analysis job kuyruğu dolu.") from error
    return JobStatusResponse.model_validate(summary.to_dict())


@router.post("/jobs/inline", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_inline(payload: InlineJobRequest, request: Request) -> JobStatusResponse:
    service = _service(request)
    if len(payload.source_code.encode("utf-8")) > service.settings.maximum_inline_source_bytes:
        raise HTTPException(status_code=413, detail="Inline source byte limiti aşıldı.")
    return _submit(
        service,
        InlinePythonSource(payload.source_code),
        payload.analysis,
    )


@router.post("/jobs/upload", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="Tek Python .py dosyası")],
    analysis: Annotated[str, Form()] = "{}",
) -> JobStatusResponse:
    service = _service(request)
    try:
        options = AnalysisOptionsRequest.model_validate_json(analysis)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail="Upload analysis configuration geçersiz.") from error
    supplied_name = file.filename or "upload.py"
    if Path(supplied_name).suffix.lower() != ".py":
        raise HTTPException(status_code=422, detail="Upload .py uzantılı olmalıdır.")
    content = await file.read(service.settings.maximum_upload_bytes + 1)
    await file.close()
    if len(content) > service.settings.maximum_upload_bytes:
        raise HTTPException(status_code=413, detail="Upload byte limiti aşıldı.")
    try:
        decoded = decode_python_source_bytes(content).text
    except PythonSourceEncodingError:
        decoded = None
    if decoded is not None and not decoded.strip():
        raise HTTPException(
            status_code=422,
            detail="Upload Python source boş bırakılamaz.",
        )
    raw_stem = Path(supplied_name).stem[:40]
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in raw_stem
    ).strip("_-") or "source"
    safe_name = f"upload_{safe_stem}.py"
    return _submit(service, UploadedPythonFile(safe_name, content), options)


@router.post("/jobs/github", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_github(payload: GitHubJobRequest, request: Request) -> JobStatusResponse:
    try:
        normalized_url, _, _ = (
            SourceAcquisitionService.validate_public_github_repository(
                str(payload.repository_url), payload.ref
            )
        )
    except SourceAcquisitionValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _submit(
        _service(request),
        PublicGitHubRepository(
            repository_url=normalized_url,
            ref=payload.ref,
        ),
        payload.analysis,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, request: Request) -> JobStatusResponse:
    try:
        return JobStatusResponse.model_validate(_service(request).get(job_id).to_dict())
    except AnalysisJobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Analysis job bulunamadı.") from error


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def job_result(job_id: str, request: Request) -> JobResultResponse:
    try:
        return JobResultResponse.model_validate(_service(request).get_result(job_id).to_dict())
    except AnalysisJobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Analysis job bulunamadı.") from error
    except AnalysisJobStateConflictError as error:
        raise HTTPException(status_code=409, detail="Analysis job sonucu henüz hazır değil.") from error


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactListResponse)
def artifacts(job_id: str, request: Request) -> ArtifactListResponse:
    try:
        values = _service(request).list_artifacts(job_id)
    except AnalysisJobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Analysis job bulunamadı.") from error
    except AnalysisJobStateConflictError as error:
        raise HTTPException(status_code=409, detail="Artifact listesi henüz hazır değil.") from error
    return ArtifactListResponse(
        job_id=job_id,
        artifacts=[ArtifactResponse.model_validate(item.to_dict()) for item in values],
    )


@router.get("/jobs/{job_id}/artifacts/{artifact_id}", response_class=FileResponse)
def download_artifact(job_id: str, artifact_id: str, request: Request) -> FileResponse:
    try:
        metadata, path = _service(request).artifact_path(job_id, artifact_id)
    except (AnalysisJobNotFoundError, AnalysisArtifactNotFoundError) as error:
        raise HTTPException(status_code=404, detail="Artifact bulunamadı.") from error
    return FileResponse(path, media_type=metadata.content_type, filename=metadata.filename)


@router.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
def cancel_job(job_id: str, request: Request) -> JobStatusResponse:
    try:
        summary = _service(request).cancel(job_id)
    except AnalysisJobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Analysis job bulunamadı.") from error
    except AnalysisJobStateConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return JobStatusResponse.model_validate(summary.to_dict())


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = _service(request)
    service.purge_expired()
    running, queued, capacity = service.capacity()
    return HealthResponse(
        status="ok",
        running_jobs=running,
        queued_jobs=queued,
        maximum_active_jobs=capacity,
    )
