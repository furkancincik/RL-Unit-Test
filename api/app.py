from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analysis_jobs import router
from api.schemas.analysis_jobs import TRUSTED_WARNING
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


def create_app(
    *,
    job_service: AnalysisJobService | None = None,
    output_root: Path = Path("output/api_jobs"),
    allowed_origins: Sequence[str] = (),
) -> FastAPI:
    """Import sırasında server veya pipeline başlatmayan application factory."""
    owned_service = job_service is None
    service = job_service or AnalysisJobService(
        settings=AnalysisJobSettings(output_root=output_root.resolve())
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owned_service:
            service.shutdown(wait=True)

    app = FastAPI(
        title="RL-Unit-Test Analysis Jobs API",
        version="1.0.0",
        description=(
            "Asynchronous external-source analysis job backend. " + TRUSTED_WARNING
        ),
        lifespan=lifespan,
    )
    app.state.analysis_job_service = service
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.include_router(router)
    return app
