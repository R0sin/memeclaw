from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class StoredIndex:
    paths: list[str]
    vectors: torch.Tensor
    model_name: str | None = None

    @property
    def total_count(self) -> int:
        return len(self.paths)


def load_index(path: Path) -> StoredIndex:
    data = torch.load(path, weights_only=False)
    return StoredIndex(
        paths=list(data["paths"]),
        vectors=data["vectors"],
        model_name=data.get("model_name"),
    )


def save_index(path: Path, index: StoredIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_file = Path(temp_path)
    try:
        torch.save({"paths": index.paths, "vectors": index.vectors, "model_name": index.model_name}, temp_file)
        os.replace(temp_file, path)
    except Exception:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
        raise


def merge_entries(
    existing: StoredIndex | None,
    new_paths: list[str],
    new_vectors: torch.Tensor,
    model_name: str,
) -> tuple[StoredIndex, int]:
    if existing is None:
        return StoredIndex(paths=new_paths, vectors=new_vectors, model_name=model_name), 0

    existing_path_set = set(existing.paths)
    new_path_set = set(new_paths)
    replaced_count = sum(1 for path in new_paths if path in existing_path_set)
    keep_indices = [idx for idx, path in enumerate(existing.paths) if path not in new_path_set]

    if keep_indices:
        kept_vectors = existing.vectors[keep_indices]
        merged_vectors = torch.cat([kept_vectors, new_vectors], dim=0)
        merged_paths = [existing.paths[idx] for idx in keep_indices] + new_paths
    else:
        merged_vectors = new_vectors
        merged_paths = list(new_paths)

    return StoredIndex(paths=merged_paths, vectors=merged_vectors, model_name=model_name), replaced_count
