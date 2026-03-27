from __future__ import annotations

import json
import os
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ENV_CONFIG_PATH = "MEMECLAW_CONFIG"
DEFAULT_TOP_K = 5
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(slots=True)
class LibraryConfig:
    image_dir: Path
    vectors_path: Path
    model: str
    exclude_dirs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_dir": str(self.image_dir),
            "model": self.model,
            "exclude_dirs": list(self.exclude_dirs),
        }


@dataclass(slots=True)
class ServerConfig:
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}


@dataclass(slots=True)
class AppConfig:
    library: LibraryConfig
    server: ServerConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "library": self.library.to_dict(),
            "server": self.server.to_dict(),
        }


def get_config_path(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()

    env_value = os.environ.get(ENV_CONFIG_PATH)
    if env_value:
        return Path(env_value).expanduser().resolve()

    return (Path.home() / ".memeclaw" / "config.toml").resolve()


def default_vectors_path(home: Path | None = None) -> Path:
    base = Path.home().resolve() if home is None else home.expanduser().resolve()
    return (base / ".memeclaw" / "vectors.pt").resolve()


def default_config() -> AppConfig:
    home = Path.home().resolve()
    image_dir = (home / ".memeclaw").resolve()
    vectors_path = default_vectors_path(home)
    return AppConfig(
        library=LibraryConfig(
            image_dir=image_dir,
            vectors_path=vectors_path,
            model="OFA-Sys/chinese-clip-vit-base-patch16",
            exclude_dirs=("thumbnails", "@eaDir", ".cache"),
        ),
        server=ServerConfig(host=DEFAULT_SERVER_HOST, port=DEFAULT_SERVER_PORT),
    )


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{name}] section")
    return value


def _require_string(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"Invalid {label}: expected a non-empty string")
    return raw.strip()


def _resolve_path(raw: Any, label: str, base_dir: Path | None = None) -> Path:
    path = Path(_require_string(raw, label)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _require_string_list(raw: Any, label: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"Invalid {label}: expected an array of strings")

    values: list[str] = []
    for index, item in enumerate(raw):
        values.append(_require_string(item, f"{label}[{index}]") )
    return tuple(values)


def _require_positive_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigError(f"Invalid {label}: expected a positive integer")
    return raw


def normalize_sub_dir(value: str | None, label: str = "sub_dir") -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        raise ConfigError(f"Invalid {label}: expected a string")

    normalized = value.strip().replace("\\", "/")
    if normalized in {"", "."}:
        return ""
    if ":" in normalized:
        raise ConfigError(f"Invalid {label}: expected a relative sub-directory")

    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigError(f"Invalid {label}: expected a relative sub-directory")

    if path.drive:
        raise ConfigError(f"Invalid {label}: expected a relative sub-directory")

    return "/".join(path.parts)


def resolve_top_k(value: int | None) -> int:
    if value is None:
        value = DEFAULT_TOP_K
    return _require_positive_int(value, "top_k")


def parse_config_dict(raw: dict[str, Any], base_dir: Path | None = None) -> AppConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a TOML object")

    library_raw = _require_section(raw, "library")
    server_raw = _require_section(raw, "server")

    resolved_base_dir = None if base_dir is None else base_dir.expanduser().resolve()

    image_dir = _resolve_path(library_raw.get("image_dir"), "library.image_dir", resolved_base_dir)
    if not image_dir.exists():
        raise ConfigError(f"Invalid library.image_dir: directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise ConfigError(f"Invalid library.image_dir: not a directory: {image_dir}")

    vectors_path = default_vectors_path()
    model = _require_string(library_raw.get("model"), "library.model")
    exclude_dirs = _require_string_list(library_raw.get("exclude_dirs"), "library.exclude_dirs")

    host = _require_string(server_raw.get("host"), "server.host")
    port = _require_positive_int(server_raw.get("port"), "server.port")
    if port > 65535:
        raise ConfigError("Invalid server.port: expected a value between 1 and 65535")

    return AppConfig(
        library=LibraryConfig(
            image_dir=image_dir,
            vectors_path=vectors_path,
            model=model,
            exclude_dirs=exclude_dirs,
        ),
        server=ServerConfig(host=host, port=port),
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = get_config_path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}. Run `memeclaw config init` first.")

    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in config file {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read config file {config_path}: {exc}") from exc

    return parse_config_dict(raw, base_dir=config_path.parent)


def render_config_toml(config: AppConfig) -> str:
    library = config.library
    server = config.server
    exclude_dirs = ", ".join(json.dumps(item, ensure_ascii=False) for item in library.exclude_dirs)
    return "\n".join(
        [
            "[library]",
            f"image_dir = {json.dumps(str(library.image_dir), ensure_ascii=False)}",
            f"model = {json.dumps(library.model, ensure_ascii=False)}",
            f"exclude_dirs = [{exclude_dirs}]",
            "",
            "[server]",
            f"host = {json.dumps(server.host, ensure_ascii=False)}",
            f"port = {server.port}",
            "",
        ]
    )


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    config_path = get_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_config_toml(config)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=config_path.parent, delete=False) as handle:
            handle.write(rendered)
            temp_path = Path(handle.name)
        temp_path.replace(config_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return config_path


def write_default_config(path: str | Path | None = None, force: bool = False) -> Path:
    config_path = get_config_path(path)
    if config_path.exists() and not force:
        raise ConfigError(f"Config file already exists: {config_path}. Use --force to overwrite it.")

    config = default_config()
    config.library.image_dir.mkdir(parents=True, exist_ok=True)
    return save_config(config, path=config_path)
