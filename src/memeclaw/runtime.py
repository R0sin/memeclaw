from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

from .config import AppConfig, ConfigError, get_config_path, load_config, normalize_sub_dir, resolve_top_k, save_config
from .indexing import build_index
from .ingest import ingest_images
from .model import Encoder, create_encoder
from .search import INDEX_MISSING_ERROR, search_stored_index
from .storage import StoredIndex, load_index

REINDEX_REQUIRED_ERROR = "Vector index must be rebuilt for the configured model. Run `memeclaw index` first."


class MemeClawRuntime:
    def __init__(
        self,
        config_path: str | Path | None = None,
        stream: TextIO | None = None,
        encoder_factory: Callable[[str, TextIO | None], Encoder] = create_encoder,
    ) -> None:
        self.config_path = get_config_path(config_path)
        self._stream = stream or sys.stderr
        self._encoder_factory = encoder_factory
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._config: AppConfig | None = None
        self._encoder: Encoder | None = None
        self._stored: StoredIndex | None = None
        self._config_mtime_ns: int | None = None
        self._index_mtime_ns: int | None = None
        self._requires_reindex = False
        self._next_index_task_id = 1
        self._index_task: dict[str, object | None] = {
            "task_id": None,
            "state": "idle",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }

    def start(self) -> None:
        with self._lock:
            self._reload_locked(force_index=True)

    def stop(self) -> None:
        with self._lock:
            self._stored = None
            self._encoder = None

    def get_config(self) -> AppConfig:
        with self._lock:
            self._refresh_locked()
            if self._config is None:
                raise ConfigError("Runtime configuration is not loaded")
            return self._config

    def get_config_dict(self) -> dict:
        return self.get_config().to_dict()

    def set_config(self, config: AppConfig) -> dict:
        with self._lock:
            save_config(config, self.config_path)
            self._reload_locked(force_index=True)
            return self._config_payload_locked()

    def reload(self) -> dict:
        with self._lock:
            self._reload_locked(force_index=True)
            payload = self._status_payload_locked()
            payload["ok"] = True
            return payload

    def status(self) -> dict:
        with self._lock:
            self._refresh_locked()
            payload = self._status_payload_locked()
            payload["ok"] = True
            return payload

    def is_ready(self) -> tuple[bool, dict]:
        with self._lock:
            try:
                self._refresh_locked()
            except Exception as exc:  # pragma: no cover - thin wrapper for health endpoints
                return False, {"ok": False, "ready": False, "error": str(exc)}

            payload = self._status_payload_locked()
            ready = bool(self._encoder is not None and self._stored is not None and not self._requires_reindex)
            payload.update({"ok": ready, "ready": ready})
            return ready, payload

    def search(self, query: str, top_k: int | None = None) -> dict:
        with self._lock:
            self._refresh_locked()
            preflight = self._preflight_search_locked()
            if preflight is not None:
                return preflight

            if self._config is None or self._encoder is None or self._stored is None:
                raise ConfigError("Runtime is not initialized")

            resolved_top_k = resolve_top_k(top_k)
            stored = self._stored
            encoder = self._encoder

        return search_stored_index(query=query, stored=stored, encoder=encoder, top_k=resolved_top_k)

    def index(self) -> dict:
        with self._write_lock:
            return self._build_index_once()

    def ingest(self, sources: list[str], sub_dir: str | None = None) -> dict:
        with self._write_lock:
            with self._lock:
                self._refresh_locked()
                if self._config is None or self._encoder is None:
                    raise ConfigError("Runtime is not initialized")
                if self._requires_reindex:
                    return {"ok": False, "error": REINDEX_REQUIRED_ERROR}

                image_dir = self._config.library.image_dir
                vectors_path = self._config.library.vectors_path
                encoder = self._encoder
                resolved_sub_dir = normalize_sub_dir(sub_dir, "sub_dir") if sub_dir is not None else ""

            result = ingest_images(
                sources=sources,
                image_dir=image_dir,
                vectors_path=vectors_path,
                encoder=encoder,
                sub_dir=resolved_sub_dir,
                stream=self._stream,
            )

            with self._lock:
                self._refresh_locked()
            return result

    def start_index_task(self) -> dict:
        with self._lock:
            self._refresh_locked()
            if self._config is None or self._encoder is None:
                raise ConfigError("Runtime is not initialized")
            if self._index_task["state"] == "running":
                payload = self._index_task_payload_locked()
                payload["ok"] = False
                payload["error"] = "Index build is already running"
                return payload

            task_id = self._next_index_task_id
            self._next_index_task_id += 1
            self._index_task = {
                "task_id": task_id,
                "state": "running",
                "started_at": self._now_iso(),
                "finished_at": None,
                "error": None,
                "result": None,
            }

        thread = threading.Thread(target=self._run_index_task, args=(task_id,), daemon=True, name=f"memeclaw-index-{task_id}")
        thread.start()

        with self._lock:
            payload = self._index_task_payload_locked()
            payload["ok"] = True
            payload["accepted"] = True
            return payload

    def index_status(self) -> dict:
        with self._lock:
            self._refresh_locked()
            payload = self._index_task_payload_locked()
            payload["ok"] = True
            return payload

    def _build_index_once(self) -> dict:
        with self._lock:
            self._refresh_locked()
            if self._config is None or self._encoder is None:
                raise ConfigError("Runtime is not initialized")

            image_dir = self._config.library.image_dir
            vectors_path = self._config.library.vectors_path
            exclude_dirs = self._config.library.exclude_dirs
            encoder = self._encoder

        result = build_index(
            image_dir=image_dir,
            vectors_path=vectors_path,
            encoder=encoder,
            exclude_dirs=exclude_dirs,
            stream=self._stream,
        )

        with self._lock:
            self._refresh_locked()
        return result

    def _run_index_task(self, task_id: int) -> None:
        try:
            with self._write_lock:
                result = self._build_index_once()
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}

        with self._lock:
            if self._index_task["task_id"] != task_id:
                return
            self._index_task["finished_at"] = self._now_iso()
            self._index_task["result"] = result
            self._index_task["error"] = None if result.get("ok") else str(result.get("error", "Index build failed"))
            self._index_task["state"] = "succeeded" if result.get("ok") else "failed"

    def _preflight_search_locked(self) -> dict | None:
        if self._config is None:
            raise ConfigError("Runtime configuration is not loaded")

        if self._requires_reindex:
            return {"ok": False, "error": REINDEX_REQUIRED_ERROR}

        if self._stored is None:
            return {
                "ok": False,
                "error": INDEX_MISSING_ERROR.format(vectors_path=self._config.library.vectors_path),
            }

        if self._stored.model_name and self._stored.model_name != self._config.library.model:
            self._requires_reindex = True
            return {"ok": False, "error": REINDEX_REQUIRED_ERROR}

        return None

    def _refresh_locked(self) -> None:
        if self._config is None:
            self._reload_locked(force_index=True)
            return

        config_mtime_ns = self._path_mtime_ns(self.config_path)
        if config_mtime_ns != self._config_mtime_ns:
            self._reload_locked(force_index=True)
            return

        if self._config is None:
            return

        index_mtime_ns = self._path_mtime_ns(self._config.library.vectors_path)
        if index_mtime_ns != self._index_mtime_ns:
            self._load_index_locked()
            self._sync_requires_reindex_locked()

    def _reload_locked(self, force_index: bool) -> None:
        config = load_config(self.config_path)
        config_mtime_ns = self._path_mtime_ns(self.config_path)
        previous = self._config
        model_changed = previous is not None and previous.library.model != config.library.model
        library_changed = (
            previous is None
            or previous.library.image_dir != config.library.image_dir
            or previous.library.exclude_dirs != config.library.exclude_dirs
        )

        if self._encoder is None or model_changed:
            self._encoder = self._encoder_factory(config.library.model, self._stream)

        self._config = config
        self._config_mtime_ns = config_mtime_ns

        if force_index or library_changed or model_changed or self._stored is None:
            self._load_index_locked()

        if model_changed:
            self._requires_reindex = True
        else:
            self._sync_requires_reindex_locked()

    def _load_index_locked(self) -> None:
        if self._config is None:
            raise ConfigError("Runtime configuration is not loaded")

        vectors_path = self._config.library.vectors_path
        self._index_mtime_ns = self._path_mtime_ns(vectors_path)
        if self._index_mtime_ns is None:
            self._stored = None
            return

        self._stored = load_index(vectors_path)

    def _sync_requires_reindex_locked(self) -> None:
        if self._config is None or self._stored is None:
            self._requires_reindex = False
            return

        if self._stored.model_name and self._stored.model_name != self._config.library.model:
            self._requires_reindex = True
            return

        self._requires_reindex = False

    def _config_payload_locked(self) -> dict:
        if self._config is None:
            raise ConfigError("Runtime configuration is not loaded")
        return self._config.to_dict()

    def _status_payload_locked(self) -> dict:
        if self._config is None:
            raise ConfigError("Runtime configuration is not loaded")

        vectors_path = self._config.library.vectors_path
        total_images = 0 if self._stored is None else self._stored.total_count
        return {
            "config_path": str(self.config_path),
            "model": self._config.library.model,
            "image_dir": str(self._config.library.image_dir),
            "vectors_path": str(vectors_path),
            "exclude_dirs": list(self._config.library.exclude_dirs),
            "server": self._config.server.to_dict(),
            "index_exists": vectors_path.exists(),
            "index_loaded": self._stored is not None,
            "index_mtime_ns": self._index_mtime_ns,
            "total_images": total_images,
            "requires_reindex": self._requires_reindex,
            "stored_model": None if self._stored is None else self._stored.model_name,
            "index_task": self._index_task_payload_locked(),
        }

    def _index_task_payload_locked(self) -> dict:
        result = self._index_task["result"]
        return {
            "task_id": self._index_task["task_id"],
            "state": self._index_task["state"],
            "running": self._index_task["state"] == "running",
            "started_at": self._index_task["started_at"],
            "finished_at": self._index_task["finished_at"],
            "error": self._index_task["error"],
            "result": None if result is None else dict(result),
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _path_mtime_ns(path: Path) -> int | None:
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
