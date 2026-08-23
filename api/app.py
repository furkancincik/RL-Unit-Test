from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from api.routes.analysis_jobs import router
from api.schemas.analysis_jobs import TRUSTED_WARNING
from services.analysis_job_service import AnalysisJobService, AnalysisJobSettings


_CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "font-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    )
)


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

    @app.middleware("http")
    async def add_security_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        return response

    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.include_router(router)
    web_root = (Path(__file__).resolve().parent.parent / "web").resolve()
    index_file = (web_root / "index.html").resolve()
    if index_file.parent != web_root:
        raise RuntimeError("Web index containment doğrulaması başarısız.")

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    def web_interface() -> FileResponse:
        return FileResponse(index_file, media_type="text/html")

    app.mount("/static", StaticFiles(directory=web_root), name="static")
    return app
