from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO

from .config import SUPPORTED_EXTS, normalize_sub_dir
from .indexing import add_images
from .model import Encoder


def choose_unique_path(dest_dir: Path, original_name: str) -> Path:
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    candidate = dest_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    counter = 1
    while True:
        suffix_part = "" if counter == 1 else f"_{counter}"
        candidate = dest_dir / f"{stem}_{timestamp}{suffix_part}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_images(
    sources: Iterable[str | Path],
    image_dir: Path,
    sub_dir: str = "",
    stream: TextIO | None = None,
) -> dict:
    log = stream or sys.stderr
    normalized_sub_dir = normalize_sub_dir(sub_dir)
    dest_dir = image_dir / normalized_sub_dir if normalized_sub_dir else image_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for source in sources:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            errors.append({"source": str(source), "error": "file does not exist"})
            continue
        if source_path.suffix.lower() not in SUPPORTED_EXTS:
            errors.append({"source": str(source), "error": f"unsupported format {source_path.suffix}"})
            continue

        dest_path = choose_unique_path(dest_dir, source_path.name)
        try:
            shutil.copy2(source_path, dest_path)
            saved.append(
                {
                    "source": str(source_path),
                    "dest": str(dest_path.resolve()),
                    "filename": dest_path.name,
                }
            )
            print(f"Saved {dest_path.name}", file=log, flush=True)
        except Exception as exc:
            errors.append({"source": str(source_path), "error": str(exc)})

    return {
        "ok": bool(saved),
        "saved_count": len(saved),
        "saved_paths": [item["dest"] for item in saved],
        "saved": saved,
        "error_count": len(errors),
        "errors": errors,
        "dest_dir": str(dest_dir.resolve()),
    }


def ingest_images(
    sources: Iterable[str | Path],
    image_dir: Path,
    vectors_path: Path,
    encoder: Encoder,
    sub_dir: str = "",
    stream: TextIO | None = None,
) -> dict:
    log = stream or sys.stderr
    save_result = copy_images(sources=sources, image_dir=image_dir, sub_dir=sub_dir, stream=log)
    if not save_result["ok"]:
        return save_result

    add_result = add_images(save_result["saved_paths"], vectors_path=vectors_path, encoder=encoder, stream=log)
    return {
        "ok": bool(save_result["ok"] and add_result["ok"]),
        "saved_count": save_result["saved_count"],
        "saved_paths": save_result["saved_paths"],
        "dest_dir": save_result["dest_dir"],
        "error_count": save_result["error_count"],
        "errors": save_result["errors"],
        "error": add_result.get("error"),
        "added_count": add_result.get("added_count", 0),
        "replaced_count": add_result.get("replaced_count", 0),
        "total_count": add_result.get("total_count", 0),
        "vector_dim": add_result.get("vector_dim"),
        "save_path": add_result.get("save_path", str(vectors_path)),
    }

