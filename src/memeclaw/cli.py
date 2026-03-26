from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx

from .config import get_config_path, load_config, parse_config_dict, save_config, write_default_config


SERVICE_UNAVAILABLE_MESSAGE = (
    "MemeClaw service is unavailable at {base_url}. Start it with `memeclaw serve` and try again."
)
DEFAULT_TIMEOUT_SECONDS = 30.0
INDEX_POLL_INTERVAL_SECONDS = 0.2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memeclaw", description="Local image semantic search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show the current service status")
    status_parser.add_argument("--json", action="store_true")

    index_parser = subparsers.add_parser("index", help="Build the full image index")
    index_parser.add_argument("--json", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search indexed images with natural language")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int)
    search_parser.add_argument("--json", action="store_true")

    ingest_parser = subparsers.add_parser("ingest", help="Copy files into the image library and index them")
    ingest_parser.add_argument("sources", nargs="+")
    ingest_parser.add_argument("--sub-dir")
    ingest_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("serve", help="Run the FastAPI service")

    config_parser = subparsers.add_parser("config", help="Manage the MemeClaw config file")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    config_init_parser = config_subparsers.add_parser("init", help="Create the default config file")
    config_init_parser.add_argument("--force", action="store_true")
    config_init_parser.add_argument("--json", action="store_true")

    config_show_parser = config_subparsers.add_parser("show", help="Show the active config")
    config_show_parser.add_argument("--json", action="store_true")

    config_validate_parser = config_subparsers.add_parser("validate", help="Validate the active config")
    config_validate_parser.add_argument("--json", action="store_true")

    config_set_parser = config_subparsers.add_parser("set", help="Update fields in the active config")
    config_set_parser.add_argument("--image-dir")
    config_set_parser.add_argument("--model")
    config_set_parser.add_argument("--exclude-dir", action="append", dest="exclude_dirs")
    config_set_parser.add_argument("--clear-exclude-dirs", action="store_true")
    config_set_parser.add_argument("--host")
    config_set_parser.add_argument("--port", type=int)
    config_set_parser.add_argument("--json", action="store_true")

    return parser


def _print_result(result: dict, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
        return

    if result.get("ok"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("error", "Unknown error"), file=sys.stderr)


def _service_base_url() -> str:
    config = load_config()
    host = config.server.host.strip()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"http://{host}:{config.server.port}"


def _parse_service_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        response.raise_for_status()
        raise ValueError("Service returned a non-JSON response")

    if isinstance(payload, dict):
        return payload

    raise ValueError("Service returned an invalid JSON payload")


def _request_service(method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = _service_base_url()
    url = f"{base_url}{path}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            response = client.request(method=method, url=url, json=json_body)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": SERVICE_UNAVAILABLE_MESSAGE.format(base_url=base_url), "details": str(exc)}

    payload = _parse_service_response(response)
    if response.is_success:
        if isinstance(payload, dict) and "ok" not in payload:
            payload = {"ok": True, **payload}
        return payload

    if isinstance(payload, dict):
        payload.setdefault("ok", False)
        return payload

    return {"ok": False, "error": f"Service request failed with status {response.status_code}"}


def _wait_for_index_task() -> dict[str, Any]:
    while True:
        status = _request_service("GET", "/v1/index")
        if not status.get("ok"):
            return status

        state = status.get("state")
        if state == "succeeded":
            result = status.get("result")
            return result if isinstance(result, dict) else {"ok": False, "error": "Index task finished without a result"}
        if state == "failed":
            result = status.get("result")
            if isinstance(result, dict):
                result.setdefault("ok", False)
                return result
            return {"ok": False, "error": status.get("error", "Index task failed")}

        time.sleep(INDEX_POLL_INTERVAL_SECONDS)


def _run_service_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "status":
        return _request_service("GET", "/v1/status")
    if args.command == "index":
        result = _request_service("POST", "/v1/index")
        if not result.get("ok"):
            return result
        if result.get("accepted"):
            return _wait_for_index_task()
        return result
    if args.command == "search":
        payload: dict[str, Any] = {"query": args.query}
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        return _request_service("POST", "/v1/search", json_body=payload)

    payload = {"source_paths": args.sources}
    if args.sub_dir is not None:
        payload["sub_dir"] = args.sub_dir
    return _request_service("POST", "/v1/ingest", json_body=payload)


def _config_update_requested(args: argparse.Namespace) -> bool:
    return any(
        [
            args.image_dir is not None,
            args.model is not None,
            args.exclude_dirs is not None,
            args.clear_exclude_dirs,
            args.host is not None,
            args.port is not None,
        ]
    )


def _run_config_command(args: argparse.Namespace) -> dict:
    config_path = get_config_path()
    if args.config_command == "init":
        created_path = write_default_config(force=args.force)
        config = load_config(created_path)
        return {"ok": True, "path": str(created_path), "config": config.to_dict()}

    if args.config_command == "set":
        if not _config_update_requested(args):
            raise ValueError("No config fields were provided. Use `memeclaw config set --help` for options.")

        config = load_config(config_path)
        raw = config.to_dict()
        library_raw = raw["library"]
        server_raw = raw["server"]

        if args.image_dir is not None:
            library_raw["image_dir"] = args.image_dir
        if args.model is not None:
            library_raw["model"] = args.model
        if args.clear_exclude_dirs:
            library_raw["exclude_dirs"] = []
        if args.exclude_dirs is not None:
            library_raw["exclude_dirs"] = args.exclude_dirs
        if args.host is not None:
            server_raw["host"] = args.host
        if args.port is not None:
            server_raw["port"] = args.port

        updated = parse_config_dict(raw, base_dir=config_path.parent)
        save_config(updated, config_path)
        return {"ok": True, "path": str(config_path), "config": updated.to_dict()}

    config = load_config(config_path)
    return {"ok": True, "path": str(config_path), "config": config.to_dict()}


def _run_serve_command() -> int:
    from .api import create_app
    import uvicorn

    config = load_config()
    uvicorn.run(create_app(), host=config.server.host, port=config.server.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_mode = getattr(args, "json", False)

    try:
        if args.command == "serve":
            return _run_serve_command()

        if args.command == "config":
            result = _run_config_command(args)
        else:
            result = _run_service_command(args)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    _print_result(result, json_mode=json_mode)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
