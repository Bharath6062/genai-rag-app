from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

VECTORS_FILE = Path("rag/out/resume_vectors.npy")
META_FILE = Path("rag/out/resume_meta.json")
EMBED_MODEL = "text-embedding-3-small"

TOP_K = 3  # show best 3 chunks


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def main():
    load_dotenv()
    client = OpenAI()

    if not VECTORS_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Missing vectors/meta. Run embed_index_openai.py first.")

    vectors = np.load(VECTORS_FILE).astype(np.float32)  # shape (N, D)
    chunks = json.loads(META_FILE.read_text(encoding="utf-8"))

    # normalize stored vectors for cosine similarity
    vectors = np.array([normalize(v) for v in vectors], dtype=np.float32)

    while True:
        q = input("\nAsk a question (or type exit): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        q_resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
        q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
        q_vec = normalize(q_vec)

        # cosine similarity = dot product (since normalized)
        scores = vectors @ q_vec
        top_idx = np.argsort(-scores)[:TOP_K]

        print("\n--- TOP MATCHES ---")
        for rank, i in enumerate(top_idx, start=1):
            print(f"\n#{rank}  score={scores[i]:.3f}  chunk={i}")
            print(chunks[i][:500])  # show first 500 chars


if __name__ == "__main__":
    main()
