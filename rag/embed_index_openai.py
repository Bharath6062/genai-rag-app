# rag/embed_index_openai.py  (MULTI-DOC)
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"

META_FILE = Path("rag/out/meta.json")
VECTORS_FILE = Path("rag/out/vectors.npy")

def main():
    load_dotenv()
    client = OpenAI()

    if not META_FILE.exists():
        print(f"❌ Missing {META_FILE}. Run ingest.py first.")
        return

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if not meta:
        print("❌ meta.json is empty. Nothing to embed.")
        return

    chunks = [m["text"] for m in meta]
    print(f"Chunks loaded: {len(chunks)}")
    print("Creating embeddings...")

    # Batch embed (safe chunk sizes)
    embeddings = []
    BATCH = 64

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    vectors = np.array(embeddings, dtype=np.float32)
    np.save(VECTORS_FILE, vectors)

    print(f"Embeddings saved to: {VECTORS_FILE}")
    print(f"Vector shape: {vectors.shape}")

if __name__ == "__main__":
    main()
