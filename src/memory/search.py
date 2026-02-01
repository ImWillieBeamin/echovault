"""Hybrid search combining FTS5 keyword search and semantic vector search."""

from typing import Optional

from memory.db import MemoryDB
from memory.embeddings.base import EmbeddingProvider


def merge_results(
    fts_results: list[dict],
    vec_results: list[dict],
    fts_weight: float = 0.3,
    vec_weight: float = 0.7,
    limit: int = 5,
) -> list[dict]:
    """Merge FTS5 and vector search results with weighted scoring.

    Args:
        fts_results: Results from FTS5 keyword search with 'id' and 'score' fields
        vec_results: Results from vector search with 'id' and 'score' fields
        fts_weight: Weight for FTS5 scores (default 0.3)
        vec_weight: Weight for vector scores (default 0.7)
        limit: Maximum number of results to return

    Returns:
        Merged and re-ranked results, sorted by combined score descending
    """
    # Normalize FTS scores to 0-1
    if fts_results:
        max_fts = max(r["score"] for r in fts_results) or 1.0
        for r in fts_results:
            r["score"] = r["score"] / max_fts if max_fts > 0 else 0.0

    # Normalize vec scores to 0-1
    if vec_results:
        max_vec = max(r["score"] for r in vec_results) or 1.0
        for r in vec_results:
            r["score"] = r["score"] / max_vec if max_vec > 0 else 0.0

    # Combine with weighted scoring, dedup by id
    scores: dict[str, dict] = {}
    for r in fts_results:
        rid = r["id"]
        scores[rid] = dict(r)
        scores[rid]["score"] = fts_weight * r["score"]
    for r in vec_results:
        rid = r["id"]
        if rid in scores:
            scores[rid]["score"] += vec_weight * r["score"]
        else:
            scores[rid] = dict(r)
            scores[rid]["score"] = vec_weight * r["score"]

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:limit]


def hybrid_search(
    db: MemoryDB,
    embedding_provider: Optional[EmbeddingProvider],
    query: str,
    limit: int = 5,
    project: Optional[str] = None,
    source: Optional[str] = None,
) -> list[dict]:
    """Run FTS5 and optionally vector search, merge results.

    When embedding_provider is None, runs FTS-only search.

    Args:
        db: Memory database instance
        embedding_provider: Embedding provider for query vectorization, or None for FTS-only
        query: Search query string
        limit: Maximum number of results to return
        project: Optional project filter
        source: Optional source filter

    Returns:
        Merged and re-ranked search results
    """
    fts_results = db.fts_search(query, limit=limit * 2, project=project, source=source)

    if embedding_provider is None:
        # FTS-only mode: normalize scores and return directly
        if fts_results:
            max_score = max(r["score"] for r in fts_results) or 1.0
            for r in fts_results:
                r["score"] = r["score"] / max_score if max_score > 0 else 0.0
        return fts_results[:limit]

    query_vec = embedding_provider.embed(query)
    vec_results = db.vector_search(
        query_vec, limit=limit * 2, project=project, source=source
    )
    return merge_results(fts_results, vec_results, limit=limit)
