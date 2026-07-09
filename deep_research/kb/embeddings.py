"""Embedding-based similarity for claim resolution (decision 25).

The step-0 spike validated this approach directly: lexical/trigram matching
caught zero real cross-source claim duplicates, but embedding cosine
similarity did (spike/FINDINGS.md). This is a candidate-generation signal only
— it feeds resolution_candidates for review, never an auto-merge (precision
degrades fast past the top few pairs per claim, per the spike's validation).
"""

import math
from dataclasses import dataclass

import httpx

from deep_research.config import Config
from deep_research.kb.db import KBDatabase


async def embed_texts(
    texts: list[str], base_url: str, model: str, instruction_prefix: str = "clustering: ",
) -> list[list[float]]:
    if not texts:
        return []
    prefixed = [f"{instruction_prefix}{t}" for t in texts]
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{base_url}/api/embed", json={"model": model, "input": prefixed})
        resp.raise_for_status()
        return resp.json()["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class BackfillResult:
    chunks_embedded: int = 0
    chunks_failed: int = 0
    claims_embedded: int = 0
    claims_failed: int = 0


async def backfill_embeddings(
    kb_db: KBDatabase, config: Config, batch_size: int = 64,
) -> BackfillResult:
    """Embeds every existing chunk/claim missing an embedding — rows that
    predate step 8, or whose write-time embedding attempt failed because
    Ollama was unreachable at the time. Idempotent and safe to re-run anytime:
    only embedding IS NULL rows are ever touched."""
    result = BackfillResult()
    base_url = config.kb.embedding_base_url
    model = config.kb.embedding_model

    chunks = await kb_db.list_chunks_missing_embedding()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vectors = await embed_texts([c["chunk_text"] for c in batch], base_url, model)
        except Exception:
            result.chunks_failed += len(batch)
            continue
        for chunk, vector in zip(batch, vectors):
            await kb_db.set_chunk_embedding(chunk["id"], vector)
            result.chunks_embedded += 1

    claims = await kb_db.list_claims_missing_embedding()
    for i in range(0, len(claims), batch_size):
        batch = claims[i : i + batch_size]
        try:
            vectors = await embed_texts([c["canonical_text"] for c in batch], base_url, model)
        except Exception:
            result.claims_failed += len(batch)
            continue
        for claim, vector in zip(batch, vectors):
            await kb_db.set_claim_embedding(claim["id"], vector)
            result.claims_embedded += 1

    return result
