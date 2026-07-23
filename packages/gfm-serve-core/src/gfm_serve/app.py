from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api import router
from .backends import ReconstructionBackend, create_backend
from .config import Settings, get_settings
from .errors import ApiError
from .logging import configure_logging


LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    backend: ReconstructionBackend | None = None,
    load_engine_on_startup: bool = True,
) -> FastAPI:
    configure_logging()
    resolved_settings = settings or get_settings()
    resolved_settings.ensure_directories()
    resolved_backend = backend or create_backend(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_engine_on_startup:
            try:
                resolved_backend.load()
            except Exception:
                LOGGER.exception("Failed to load reconstruction backend at startup")
        yield

    app = FastAPI(title=resolved_settings.service_name, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.backend = resolved_backend

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message}},
            status_code=exc.status_code,
        )

    app.include_router(router)
    return app
