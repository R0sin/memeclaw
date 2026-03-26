from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .model import Encoder
from .storage import StoredIndex, load_index


MODEL_MISMATCH_ERROR = "Vector index model does not match the configured model. Run `memeclaw index` first."
INDEX_MISSING_ERROR = "Vector index not found: {vectors_path}. Run `memeclaw index` first."


def _search_stored(query: str, stored: StoredIndex, encoder: Encoder, top_k: int) -> dict:
    if stored.model_name and stored.model_name != encoder.model_name:
        return {"ok": False, "error": MODEL_MISMATCH_ERROR}

    if top_k <= 0:
        return {"ok": False, "error": "top_k must be greater than 0"}

    if not stored.paths:
        return {"ok": False, "error": "Vector index is empty. Run `memeclaw index` first."}

    text_vector = encoder.encode_text(query)
    similarities = (text_vector @ stored.vectors.T).squeeze(0)
    limit = min(top_k, len(stored.paths))
    scores, indices = similarities.topk(limit)

    results = [
        {
            "rank": offset + 1,
            "score": round(float(score), 4),
            "path": stored.paths[int(idx)],
            "filename": Path(stored.paths[int(idx)]).name,
        }
        for offset, (score, idx) in enumerate(zip(scores, indices))
    ]

    return {
        "ok": True,
        "query": query,
        "total_images": len(stored.paths),
        "results": results,
    }


def search_stored_index(query: str, stored: StoredIndex, encoder: Encoder, top_k: int) -> dict:
    return _search_stored(query=query, stored=stored, encoder=encoder, top_k=top_k)


def search_index(
    query: str,
    vectors_path: Path,
    encoder: Encoder,
    top_k: int,
    stream: TextIO | None = None,
) -> dict:
    if not vectors_path.exists():
        return {
            "ok": False,
            "error": INDEX_MISSING_ERROR.format(vectors_path=vectors_path),
        }

    stored = load_index(vectors_path)
    return _search_stored(query=query, stored=stored, encoder=encoder, top_k=top_k)
