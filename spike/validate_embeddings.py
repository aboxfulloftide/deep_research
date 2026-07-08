"""Validate whether embedding similarity catches cross-source claim duplicates
that lexical/trigram matching missed (spike/FINDINGS.md, step 0 -> step 1 handoff).

Throwaway. Embeds every claim from spike/output/observations.jsonl with
nomic-embed-text (already running locally via Ollama), then reports the
highest-similarity cross-source claim pairs for manual judgment.
"""

import json
import math
from pathlib import Path

import httpx

OBSERVATIONS_PATH = Path(__file__).parent / "output" / "observations.jsonl"
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text:v1.5"
TOP_N = 25


def embed(texts: list[str]) -> list[list[float]]:
    prefixed = [f"clustering: {t}" for t in texts]
    with httpx.Client(timeout=120) as client:
        resp = client.post(OLLAMA_URL, json={"model": EMBED_MODEL, "input": prefixed})
        resp.raise_for_status()
        return resp.json()["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main():
    claims = [json.loads(line) for line in open(OBSERVATIONS_PATH)]
    print(f"Loaded {len(claims)} claims")

    texts = [c["claim_text"] for c in claims]
    print("Embedding all claims...")
    vectors = embed(texts)
    print(f"Got {len(vectors)} vectors of dim {len(vectors[0])}")

    pairs = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if claims[i]["source"] == claims[j]["source"]:
                continue
            sim = cosine(vectors[i], vectors[j])
            pairs.append((sim, i, j))

    pairs.sort(reverse=True)

    print(f"\nTop {TOP_N} cross-source claim pairs by cosine similarity:\n")
    for sim, i, j in pairs[:TOP_N]:
        print(f"{sim:.3f}")
        print(f"  [article]   {claims[i]['claim_text'] if claims[i]['source']=='article' else claims[j]['claim_text']}")
        print(f"  [transcript] {claims[j]['claim_text'] if claims[i]['source']=='article' else claims[i]['claim_text']}")
        print()

    sims = [p[0] for p in pairs]
    sims.sort()
    n = len(sims)
    print(f"\nCross-source pair count: {n}")
    print(f"similarity distribution: min={sims[0]:.3f} p50={sims[n//2]:.3f} p90={sims[int(n*0.9)]:.3f} max={sims[-1]:.3f}")

    for thresh in (0.75, 0.8, 0.85, 0.9):
        count = sum(1 for s in sims if s >= thresh)
        print(f"pairs >= {thresh}: {count}")


if __name__ == "__main__":
    main()
