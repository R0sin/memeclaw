from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from datetime import datetime
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
import torch
from PIL import Image

from memeclaw import cli
from memeclaw.config import (
    AppConfig,
    ConfigError,
    ENV_CONFIG_PATH,
    LibraryConfig,
    ServerConfig,
    default_vectors_path,
    get_config_path,
    load_config,
    parse_config_dict,
    save_config,
    write_default_config,
)
from memeclaw.indexing import add_images, build_index
from memeclaw.ingest import copy_images, ingest_images
from memeclaw.runtime import MemeClawRuntime
from memeclaw.search import search_index
from memeclaw.storage import load_index

try:
    from fastapi.testclient import TestClient
    from memeclaw.api import create_app
except Exception:  # pragma: no cover - dependency availability varies in local envs
    TestClient = None
    create_app = None


class StubEncoder:
    def __init__(self, model_name: str = "stub-model") -> None:
        self.model_name = model_name

    def encode_images(self, images):
        rows = []
        for image in images:
            red, green, blue = image.resize((1, 1)).getpixel((0, 0))
            row = torch.tensor([[float(red), float(green), float(blue)]], dtype=torch.float32)
            row = row / row.norm(dim=-1, keepdim=True)
            rows.append(row)
        return torch.cat(rows, dim=0)

    def encode_text(self, text):
        lowered = text.lower()
        if "red" in lowered:
            row = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        elif "blue" in lowered:
            row = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        else:
            row = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        return row / row.norm(dim=-1, keepdim=True)


class CountingEncoderFactory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, model_name: str, stream=None) -> StubEncoder:
        self.calls.append(model_name)
        return StubEncoder(model_name=model_name)


class DummyResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.is_success:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("GET", "http://testserver"),
                response=httpx.Response(self.status_code),
            )


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color).save(path)


def make_png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_vectors_path(root: Path) -> Path:
    return (root / ".memeclaw" / "vectors.pt").resolve()


def patch_default_vectors_path(root: Path):
    return patch("memeclaw.config.default_vectors_path", return_value=make_vectors_path(root))


def make_config(root: Path, model: str = "stub-model", host: str = "127.0.0.1", port: int = 8000) -> AppConfig:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        library=LibraryConfig(
            image_dir=image_dir,
            vectors_path=make_vectors_path(root),
            model=model,
            exclude_dirs=("cache",),
        ),
        server=ServerConfig(host=host, port=port),
    )


def save_test_config(root: Path, model: str = "stub-model", host: str = "127.0.0.1", port: int = 8000) -> tuple[Path, AppConfig]:
    config = make_config(root, model=model, host=host, port=port)
    config_path = root / "config.toml"
    save_config(config, config_path)
    return config_path, config


def wait_for_index_completion(client: TestClient, *, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/v1/index")
        payload = response.json()
        if payload["state"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for index task to finish")


class ConfigTests(unittest.TestCase):
    def test_get_config_path_uses_env_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "custom.toml"
            with patch.dict(os.environ, {ENV_CONFIG_PATH: str(custom)}, clear=True):
                self.assertEqual(get_config_path(), custom.resolve())

    def test_write_default_config_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            with patch_default_vectors_path(Path(temp_dir)):
                written_path = write_default_config(config_path)
                self.assertEqual(written_path, config_path.resolve())
                loaded = load_config(config_path)
            self.assertTrue(loaded.library.image_dir.exists())
            self.assertEqual(loaded.server.port, 8000)
            self.assertEqual(loaded.library.vectors_path, make_vectors_path(Path(temp_dir)))

    def test_parse_config_dict_rejects_missing_image_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = make_config(root).to_dict()
            raw["library"]["image_dir"] = str(root / "missing")
            with self.assertRaisesRegex(ConfigError, "directory not found"):
                parse_config_dict(raw)

    def test_load_config_resolves_relative_paths_from_config_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "settings"
            image_dir = config_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.toml"
            config_path.write_text(
                '\n'.join(
                    [
                        '[library]',
                        'image_dir = "./images"',
                        'vectors_path = "./legacy-vectors.pt"',
                        'model = "stub-model"',
                        'exclude_dirs = ["cache"]',
                        '',
                        '[server]',
                        'host = "127.0.0.1"',
                        'port = 8000',
                        '',
                    ]
                ),
                encoding='utf-8',
            )

            cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch_default_vectors_path(root):
                    loaded = load_config(config_path)
            finally:
                os.chdir(cwd)

            self.assertEqual(loaded.library.image_dir, image_dir.resolve())
            self.assertEqual(loaded.library.vectors_path, make_vectors_path(root))

    def test_parse_config_dict_rejects_bool_for_positive_int_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = make_config(root).to_dict()
            raw['server']['port'] = True
            with self.assertRaisesRegex(ConfigError, 'server.port'):
                parse_config_dict(raw)

    def test_parse_config_dict_allows_legacy_ingest_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = make_config(root).to_dict()
            raw['search'] = {'default_top_k': 99}
            raw['ingest'] = {'default_sub_dir': 'legacy'}
            raw['library']['vectors_path'] = str(root / 'legacy-vectors.pt')
            with patch_default_vectors_path(root):
                parsed = parse_config_dict(raw)
            self.assertEqual(parsed.to_dict(), make_config(root).to_dict())
            self.assertEqual(parsed.library.vectors_path, make_vectors_path(root))

    def test_load_config_wraps_os_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.toml'
            with patch_default_vectors_path(Path(temp_dir)):
                write_default_config(config_path)
                with patch('pathlib.Path.open', side_effect=PermissionError('denied')):
                    with self.assertRaisesRegex(ConfigError, 'Unable to read config file'):
                        load_config(config_path)


class IndexWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_dir = self.root / "images"
        self.vectors_path = make_vectors_path(self.root)
        self.vectors_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = StubEncoder()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_storage_compatibility_with_legacy_pt_format(self):
        torch.save(
            {
                "paths": [str((self.root / "legacy.jpg").resolve())],
                "vectors": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            },
            self.vectors_path,
        )
        loaded = load_index(self.vectors_path)
        self.assertIsNone(loaded.model_name)
        self.assertEqual(loaded.total_count, 1)

    def test_index_search_add_and_ingest(self):
        red = self.image_dir / "red.png"
        blue = self.image_dir / "blue.png"
        skipped = self.image_dir / "skip.txt"
        excluded = self.image_dir / "cache" / "hidden.png"
        make_image(red, (255, 0, 0))
        make_image(blue, (0, 0, 255))
        make_image(excluded, (0, 255, 0))
        skipped.write_text("not an image", encoding="utf-8")

        index_result = build_index(
            image_dir=self.image_dir,
            vectors_path=self.vectors_path,
            encoder=self.encoder,
            exclude_dirs=("cache",),
        )
        self.assertTrue(index_result["ok"])
        self.assertEqual(index_result["image_count"], 2)
        self.assertEqual(load_index(self.vectors_path).model_name, "stub-model")

        search_result = search_index(
            query="red",
            vectors_path=self.vectors_path,
            encoder=self.encoder,
            top_k=1,
        )
        self.assertTrue(search_result["ok"])
        self.assertEqual(search_result["results"][0]["filename"], "red.png")

        add_result_first = add_images([str(red)], self.vectors_path, self.encoder)
        add_result_second = add_images([str(red)], self.vectors_path, self.encoder)
        self.assertTrue(add_result_first["ok"])
        self.assertTrue(add_result_second["ok"])
        self.assertEqual(add_result_second["total_count"], 2)
        self.assertEqual(add_result_second["replaced_count"], 1)

        ingest_source = self.root / "upload-source.png"
        make_image(ingest_source, (0, 255, 0))
        ingest_result = ingest_images(
            sources=[str(ingest_source)],
            image_dir=self.image_dir,
            vectors_path=self.vectors_path,
            encoder=self.encoder,
            sub_dir="uploads",
        )
        self.assertTrue(ingest_result["ok"])
        self.assertEqual(ingest_result["saved_count"], 1)
        self.assertEqual(ingest_result["added_count"], 1)
        self.assertEqual(load_index(self.vectors_path).total_count, 3)

    def test_copy_images_keeps_duplicate_filenames_unique(self):
        first = self.root / "incoming-a" / "same.png"
        second = self.root / "incoming-b" / "same.png"
        make_image(first, (255, 0, 0))
        make_image(second, (0, 0, 255))

        with patch("memeclaw.ingest.datetime") as mocked_datetime:
            mocked_datetime.now.return_value = datetime(2026, 3, 26, 15, 0, 0)
            result = copy_images([str(first), str(second), str(second)], image_dir=self.image_dir, sub_dir="uploads")

        self.assertTrue(result["ok"])
        self.assertEqual(result["saved_count"], 3)
        self.assertEqual(len(set(result["saved_paths"])), 3)
        saved_names = [Path(path).name for path in result["saved_paths"]]
        self.assertEqual(len(set(saved_names)), 3)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.vectors_path_patcher = patch_default_vectors_path(self.root)
        self.vectors_path_patcher.start()
        self.addCleanup(self.vectors_path_patcher.stop)
        self.config_path, self.config = save_test_config(self.root)
        self.factory = CountingEncoderFactory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runtime_reloads_index_after_external_update(self):
        make_image(self.config.library.image_dir / "red.png", (255, 0, 0))
        runtime = MemeClawRuntime(config_path=self.config_path, encoder_factory=self.factory, stream=io.StringIO())
        runtime.start()
        try:
            self.assertTrue(runtime.index()["ok"])
            make_image(self.config.library.image_dir / "blue.png", (0, 0, 255))
            build_index(
                image_dir=self.config.library.image_dir,
                vectors_path=self.config.library.vectors_path,
                encoder=StubEncoder(),
                exclude_dirs=self.config.library.exclude_dirs,
                stream=io.StringIO(),
            )
            result = runtime.search("blue", top_k=1)
            self.assertTrue(result["ok"])
            self.assertEqual(result["results"][0]["filename"], "blue.png")
            self.assertEqual(self.factory.calls, ["stub-model"])
        finally:
            runtime.stop()

    def test_runtime_marks_reindex_when_model_changes(self):
        make_image(self.config.library.image_dir / "red.png", (255, 0, 0))
        runtime = MemeClawRuntime(config_path=self.config_path, encoder_factory=self.factory, stream=io.StringIO())
        runtime.start()
        try:
            self.assertTrue(runtime.index()["ok"])
            save_config(make_config(self.root, model="other-model"), self.config_path)
            status = runtime.status()
            self.assertTrue(status["requires_reindex"])
            search_result = runtime.search("red", top_k=1)
            self.assertFalse(search_result["ok"])
            self.assertIn("rebuilt", search_result["error"])
            self.assertEqual(self.factory.calls, ["stub-model", "other-model"])
        finally:
            runtime.stop()


class CliSmokeTests(unittest.TestCase):
    def test_cli_config_init_show_and_validate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            with patch.dict(os.environ, {ENV_CONFIG_PATH: str(config_path)}):
                with patch_default_vectors_path(Path(temp_dir)):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = cli.main(["config", "init", "--json"])
                    self.assertEqual(exit_code, 0)
                    self.assertTrue(config_path.exists())
                    created = json.loads(stdout.getvalue())
                    self.assertEqual(created["path"], str(config_path.resolve()))

                    stdout = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                        exit_code = cli.main(["config", "show", "--json"])
                    self.assertEqual(exit_code, 0)
                    shown = json.loads(stdout.getvalue())
                    self.assertIn("library", shown["config"])

                    stdout = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                        exit_code = cli.main(["config", "validate", "--json"])
                    self.assertEqual(exit_code, 0)
                    validated = json.loads(stdout.getvalue())
                    self.assertEqual(validated["path"], str(config_path.resolve()))

    def test_cli_config_set_updates_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _ = save_test_config(root)
            updated_image_dir = root / "updated-images"
            updated_image_dir.mkdir()

            with patch.dict(os.environ, {ENV_CONFIG_PATH: str(config_path)}):
                with patch_default_vectors_path(root):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                        exit_code = cli.main(
                            [
                                "config",
                                "set",
                                "--image-dir",
                                str(updated_image_dir),
                                "--port",
                                "8010",
                                "--clear-exclude-dirs",
                                "--json",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["config"]["library"]["image_dir"], str(updated_image_dir.resolve()))
            self.assertEqual(payload["config"]["server"]["port"], 8010)
            self.assertEqual(payload["config"]["library"]["exclude_dirs"], [])

    def test_cli_proxies_service_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _ = save_test_config(root)
            captured: list[tuple[str, str, dict | None]] = []

            def fake_request(method: str, path: str, *, json_body=None):
                captured.append((method, path, json_body))
                if method == "POST" and path == "/v1/index":
                    return {"ok": True, "accepted": True, "task_id": 1, "state": "running"}
                if method == "GET" and path == "/v1/index":
                    return {"ok": True, "state": "succeeded", "result": {"ok": True, "image_count": 2}}
                return {"ok": True, "path": path}

            with patch.dict(os.environ, {ENV_CONFIG_PATH: str(config_path)}):
                with patch_default_vectors_path(root):
                    with patch("memeclaw.cli._request_service", side_effect=fake_request):
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            self.assertEqual(cli.main(["status", "--json"]), 0)
                            self.assertEqual(cli.main(["index", "--json"]), 0)
                            self.assertEqual(cli.main(["search", "red", "--top-k", "3", "--json"]), 0)
                            self.assertEqual(cli.main(["ingest", "a.png", "b.png", "--sub-dir", "uploads", "--json"]), 0)

            self.assertEqual(
                captured,
                [
                    ("GET", "/v1/status", None),
                    ("POST", "/v1/index", None),
                    ("GET", "/v1/index", None),
                    ("POST", "/v1/search", {"query": "red", "top_k": 3}),
                    ("POST", "/v1/ingest", {"source_paths": ["a.png", "b.png"], "sub_dir": "uploads"}),
                ],
            )

    def test_cli_fails_when_config_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.toml"
            with patch.dict(os.environ, {ENV_CONFIG_PATH: str(missing)}, clear=True):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cli.main(["index", "--json"])
                self.assertEqual(exit_code, 1)
                self.assertIn("Config file not found", stdout.getvalue())

    def test_request_service_reports_unavailable_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _ = save_test_config(root, host="0.0.0.0", port=9001)

            class RaisingClient:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def request(self, method, url, json=None):
                    raise httpx.ConnectError("boom", request=httpx.Request(method, url))

            with patch.dict(os.environ, {ENV_CONFIG_PATH: str(config_path)}):
                with patch_default_vectors_path(root):
                    with patch("memeclaw.cli.httpx.Client", RaisingClient):
                        result = cli._request_service("POST", "/v1/search", json_body={"query": "red"})

            self.assertFalse(result["ok"])
            self.assertIn("http://127.0.0.1:9001", result["error"])


@unittest.skipIf(TestClient is None or create_app is None, "FastAPI test client unavailable")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.vectors_path_patcher = patch_default_vectors_path(self.root)
        self.vectors_path_patcher.start()
        self.addCleanup(self.vectors_path_patcher.stop)
        self.config_path, self.config = save_test_config(self.root)
        self.factory = CountingEncoderFactory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_api_config_index_search_reload_and_ingest(self):
        make_image(self.config.library.image_dir / "red.png", (255, 0, 0))
        runtime = MemeClawRuntime(config_path=self.config_path, encoder_factory=self.factory, stream=io.StringIO())
        with TestClient(create_app(runtime=runtime)) as client:
            config_response = client.get("/v1/config")
            self.assertEqual(config_response.status_code, 200)
            self.assertEqual(config_response.json()["library"]["model"], "stub-model")

            index_response = client.post("/v1/index")
            self.assertEqual(index_response.status_code, 202)
            self.assertTrue(index_response.json()["accepted"])

            index_status = wait_for_index_completion(client)
            self.assertEqual(index_status["state"], "succeeded")
            self.assertEqual(index_status["result"]["image_count"], 1)

            search_response = client.post("/v1/search", json={"query": "red", "top_k": 1})
            self.assertEqual(search_response.status_code, 200)
            self.assertEqual(search_response.json()["results"][0]["filename"], "red.png")

            reload_response = client.post("/v1/reload")
            self.assertEqual(reload_response.status_code, 200)
            self.assertTrue(reload_response.json()["index_loaded"])

            ingest_response = client.post(
                "/v1/ingest",
                data={"sub_dir": "uploads"},
                files=[("files", ("green.png", make_png_bytes((0, 255, 0)), "image/png"))],
            )
            self.assertEqual(ingest_response.status_code, 200)
            self.assertEqual(ingest_response.json()["saved_count"], 1)

    def test_api_ingest_accepts_files_array_and_keeps_duplicate_uploads(self):
        runtime = MemeClawRuntime(config_path=self.config_path, encoder_factory=self.factory, stream=io.StringIO())
        with TestClient(create_app(runtime=runtime)) as client:
            ingest_response = client.post(
                "/v1/ingest",
                data={"sub_dir": "uploads"},
                files=[
                    ("files[]", ("same.png", make_png_bytes((255, 0, 0)), "image/png")),
                    ("files[]", ("same.png", make_png_bytes((0, 0, 255)), "image/png")),
                ],
            )

        self.assertEqual(ingest_response.status_code, 200)
        payload = ingest_response.json()
        self.assertEqual(payload["saved_count"], 2)
        self.assertEqual(len(set(payload["saved_paths"])), 2)

        saved_colors = set()
        for saved_path in payload["saved_paths"]:
            with Image.open(saved_path) as image:
                saved_colors.add(image.resize((1, 1)).getpixel((0, 0)))

        self.assertEqual(saved_colors, {(255, 0, 0), (0, 0, 255)})

    def test_put_config_marks_reindex(self):
        make_image(self.config.library.image_dir / "red.png", (255, 0, 0))
        runtime = MemeClawRuntime(config_path=self.config_path, encoder_factory=self.factory, stream=io.StringIO())
        with TestClient(create_app(runtime=runtime)) as client:
            self.assertEqual(client.post("/v1/index").status_code, 202)
            self.assertEqual(wait_for_index_completion(client)["state"], "succeeded")
            payload = client.get("/v1/config").json()
            payload["library"]["model"] = "other-model"
            put_response = client.put("/v1/config", json=payload)
            self.assertEqual(put_response.status_code, 200)
            self.assertEqual(put_response.json()["library"]["model"], "other-model")

            status_response = client.get("/v1/status")
            self.assertEqual(status_response.status_code, 200)
            self.assertTrue(status_response.json()["requires_reindex"])

            ready_response = client.get("/readyz")
            self.assertEqual(ready_response.status_code, 503)

            search_response = client.post("/v1/search", json={"query": "red"})
            self.assertEqual(search_response.status_code, 409)
            self.assertIn("rebuilt", search_response.json()["error"])













