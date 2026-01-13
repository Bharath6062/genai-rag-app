from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

OUT_DIR = Path("rag/out")
META_FILE = OUT_DIR / "meta.json"
VECTORS_FILE = OUT_DIR / "vectors.npy"

EMBED_MODEL = "text-embedding-3-small"
BATCH = 128


def main():
    load_dotenv()
    client = OpenAI()

    if not META_FILE.exists():
        raise RuntimeError(f"Missing {META_FILE}. Run ingest.py first.")

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    texts = [m["text"] for m in meta]
    if not texts:
        raise RuntimeError("meta.json empty texts")

    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        print(f"Embedded {min(i + BATCH, len(texts))}/{len(texts)}")

    arr = np.array(vectors, dtype=np.float32)
    VECTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_FILE, arr)
    print(f"OK: wrote {arr.shape} -> {VECTORS_FILE}")


if __name__ == "__main__":
    main()
