"""Embedding-based similarity for claim resolution (decision 25).

The step-0 spike validated this approach directly: lexical/trigram matching
caught zero real cross-source claim duplicates, but embedding cosine
similarity did (spike/FINDINGS.md). This is a candidate-generation signal only
— it feeds resolution_candidates for review, never an auto-merge (precision
degrades fast past the top few pairs per claim, per the spike's validation).
"""

import math

import httpx


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
