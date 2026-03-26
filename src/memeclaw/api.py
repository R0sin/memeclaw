from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

from .config import AppConfig, ConfigError, parse_config_dict
from .ingest import choose_unique_path
from .runtime import MemeClawRuntime


class LibraryConfigPayload(BaseModel):
    image_dir: str
    model: str
    exclude_dirs: list[str] = Field(default_factory=list)


class ServerConfigPayload(BaseModel):
    host: str
    port: int


class AppConfigPayload(BaseModel):
    library: LibraryConfigPayload
    server: ServerConfigPayload


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class IngestPathsRequest(BaseModel):
    source_paths: list[str]
    sub_dir: str | None = None


class ApiError(Exception):
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(payload.get("error", "API error"))


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _result_status_code(result: dict[str, Any]) -> int:
    error = str(result.get("error", "")).lower()
    if "already running" in error:
        return 409
    if "run `memeclaw index` first" in error or "must be rebuilt" in error or "does not match" in error:
        return 409
    return 422


def _parse_config_payload(payload: AppConfigPayload) -> AppConfig:
    return parse_config_dict(_model_dump(payload))


def create_app(runtime: MemeClawRuntime | None = None) -> FastAPI:
    app_runtime = runtime or MemeClawRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_runtime.start()
        app.state.runtime = app_runtime
        yield
        app_runtime.stop()

    app = FastAPI(title="MemeClaw API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.payload)

    @app.exception_handler(ConfigError)
    async def handle_config_error(request: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"ok": False, "error": str(exc)})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "memeclaw", "config_path": str(app_runtime.config_path)}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready, payload = app_runtime.is_ready()
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/v1/config")
    async def get_config() -> dict[str, Any]:
        return app_runtime.get_config_dict()

    @app.put("/v1/config")
    async def put_config(payload: AppConfigPayload) -> dict[str, Any]:
        config = _parse_config_payload(payload)
        return app_runtime.set_config(config)

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        return app_runtime.status()

    @app.post("/v1/search")
    async def search(payload: SearchRequest) -> dict[str, Any]:
        result = app_runtime.search(query=payload.query, top_k=payload.top_k)
        if not result.get("ok"):
            raise ApiError(_result_status_code(result), result)
        return result

    @app.post("/v1/index")
    async def index() -> JSONResponse:
        result = app_runtime.start_index_task()
        if not result.get("ok"):
            raise ApiError(_result_status_code(result), result)
        return JSONResponse(status_code=202, content=result)

    @app.get("/v1/index")
    async def index_status() -> dict[str, Any]:
        return app_runtime.index_status()

    @app.post("/v1/reload")
    async def reload_runtime() -> dict[str, Any]:
        result = app_runtime.reload()
        if not result.get("ok"):
            raise ApiError(_result_status_code(result), result)
        return result

    @app.post("/v1/ingest")
    async def ingest(request: Request) -> dict[str, Any]:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            data = await request.json()
            payload = IngestPathsRequest.model_validate(data) if hasattr(IngestPathsRequest, "model_validate") else IngestPathsRequest.parse_obj(data)
            result = app_runtime.ingest(sources=payload.source_paths, sub_dir=payload.sub_dir)
            if not result.get("ok"):
                raise ApiError(_result_status_code(result), result)
            return result

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload_items = [*form.getlist("files"), *form.getlist("files[]")]
            uploads = [item for item in upload_items if isinstance(item, StarletteUploadFile)]
            if not uploads:
                raise ApiError(422, {"ok": False, "error": "No files were uploaded"})

            sub_dir = form.get("sub_dir")
            try:
                with tempfile.TemporaryDirectory(prefix="memeclaw-upload-") as temp_dir:
                    saved_paths: list[str] = []
                    for upload in uploads:
                        filename = Path(upload.filename or "upload.bin").name
                        dest_path = choose_unique_path(Path(temp_dir), filename)
                        with dest_path.open("wb") as handle:
                            shutil.copyfileobj(upload.file, handle)
                        saved_paths.append(str(dest_path))

                    result = app_runtime.ingest(sources=saved_paths, sub_dir=sub_dir)
                    if not result.get("ok"):
                        raise ApiError(_result_status_code(result), result)
                    return result
            finally:
                for upload in uploads:
                    await upload.close()

        raise ApiError(422, {"ok": False, "error": "Unsupported content type"})

    return app



