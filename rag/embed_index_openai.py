from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ---------- CONFIG ----------
CHUNKS_FILE = Path("rag/out/resume_chunks.txt")
VECTORS_FILE = Path("rag/out/resume_vectors.npy")
META_FILE = Path("rag/out/resume_meta.json")
EMBED_MODEL = "text-embedding-3-small"
# ----------------------------

def load_chunks():
    text = CHUNKS_FILE.read_text(encoding="utf-8")
    parts = text.split("--- CHUNK ")
    chunks = []

    for p in parts:
        p = p.strip()
        if not p:
            continue
        body = p.split("---\n", 1)[1]
        chunks.append(body.strip())

    return chunks

def main():
    load_dotenv()
    client = OpenAI()

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("resume_chunks.txt not found. Run ingest.py first.")

    chunks = load_chunks()
    print("Chunks loaded:", len(chunks))

    # Safety guard (protect your $5)
    if len(chunks) > 100:
        raise RuntimeError("Too many chunks. Something is wrong.")

    print("Creating embeddings...")

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks
    )

    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)

    np.save(VECTORS_FILE, vectors)
    META_FILE.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print("Embeddings saved to:", VECTORS_FILE)
    print("Vector shape:", vectors.shape)

if __name__ == "__main__":
    main()
