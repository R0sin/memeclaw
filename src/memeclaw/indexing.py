from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, TextIO

import torch
from PIL import Image

from .config import SUPPORTED_EXTS
from .model import Encoder
from .storage import StoredIndex, load_index, merge_entries, save_index


MODEL_MISMATCH_ERROR = "Vector index model does not match the configured model. Run `memeclaw index` first."


def scan_images(root: Path, exclude_dirs: Iterable[str]) -> list[Path]:
    exclude = set(exclude_dirs)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [entry for entry in dirnames if entry not in exclude]
        for filename in filenames:
            if Path(filename).suffix.lower() in SUPPORTED_EXTS:
                found.append((Path(dirpath) / filename).resolve())
    return found


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_index(
    image_dir: Path,
    vectors_path: Path,
    encoder: Encoder,
    exclude_dirs: Iterable[str] = (),
    stream: TextIO | None = None,
) -> dict:
    log = stream or sys.stderr
    if not image_dir.exists():
        return {"ok": False, "error": f"Image directory not found: {image_dir}"}
    if not image_dir.is_dir():
        return {"ok": False, "error": f"Image directory is not a directory: {image_dir}"}

    image_paths = scan_images(image_dir, exclude_dirs)

    if exclude_dirs:
        print(f"Excluded directories: {', '.join(sorted(exclude_dirs))}", file=log, flush=True)

    if not image_paths:
        return {"ok": False, "error": f"No supported images found in {image_dir}"}

    print(f"Found {len(image_paths)} image(s)", file=log, flush=True)

    stored_paths: list[str] = []
    vectors: list[torch.Tensor] = []
    skipped = 0

    for index, image_path in enumerate(image_paths, start=1):
        try:
            image = _open_image(image_path)
        except Exception as exc:  # pragma: no cover - rare PIL failure shape
            print(f"Skip {image_path.name}: {exc}", file=log, flush=True)
            skipped += 1
            continue

        vector = encoder.encode_images([image])
        stored_paths.append(str(image_path))
        vectors.append(vector)
        print(f"[{index}/{len(image_paths)}] {image_path.name}", file=log, flush=True)

    if not vectors:
        return {"ok": False, "error": "No images were successfully processed"}

    matrix = torch.cat(vectors, dim=0)
    save_index(vectors_path, StoredIndex(paths=stored_paths, vectors=matrix, model_name=encoder.model_name))
    return {
        "ok": True,
        "image_count": len(stored_paths),
        "skipped": skipped,
        "vector_dim": int(matrix.shape[1]),
        "save_path": str(vectors_path),
        "image_dir": str(image_dir),
        "model": encoder.model_name,
    }


def add_images(
    image_paths: Iterable[str | Path],
    vectors_path: Path,
    encoder: Encoder,
    stream: TextIO | None = None,
) -> dict:
    log = stream or sys.stderr
    existing = load_index(vectors_path) if vectors_path.exists() else None
    if existing is None:
        print("Vector index does not exist yet; creating a new one", file=log, flush=True)
    else:
        if existing.model_name and existing.model_name != encoder.model_name:
            return {"ok": False, "error": MODEL_MISMATCH_ERROR, "skipped": 0}
        print(f"Loaded existing index with {existing.total_count} image(s)", file=log, flush=True)

    deduped_inputs: dict[str, Path] = {}
    for raw_path in image_paths:
        resolved = Path(raw_path).expanduser().resolve()
        deduped_inputs[str(resolved)] = resolved

    new_paths: list[str] = []
    new_vectors: list[torch.Tensor] = []
    skipped = 0

    for image_path in deduped_inputs.values():
        if not image_path.exists():
            print(f"Skip missing file: {image_path}", file=log, flush=True)
            skipped += 1
            continue

        try:
            image = _open_image(image_path)
        except Exception as exc:
            print(f"Skip {image_path.name}: {exc}", file=log, flush=True)
            skipped += 1
            continue

        vector = encoder.encode_images([image])
        new_paths.append(str(image_path))
        new_vectors.append(vector)
        print(f"Indexed {image_path.name}", file=log, flush=True)

    if not new_vectors:
        return {"ok": False, "error": "No images were successfully processed", "skipped": skipped}

    new_matrix = torch.cat(new_vectors, dim=0)
    merged_index, replaced_count = merge_entries(existing, new_paths, new_matrix, encoder.model_name)
    save_index(vectors_path, merged_index)

    return {
        "ok": True,
        "added_count": len(new_paths),
        "replaced_count": replaced_count,
        "skipped": skipped,
        "total_count": merged_index.total_count,
        "vector_dim": int(merged_index.vectors.shape[1]),
        "save_path": str(vectors_path),
    }
