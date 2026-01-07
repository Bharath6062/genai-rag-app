# rag/answer_resume.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# -------- Paths (match what you already created) --------
VECTORS_FILE = Path("rag/out/resume_vectors.npy")
META_FILE = Path("rag/out/resume_meta.json")

# -------- Models --------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # cheap + strong for RAG

# -------- Retrieval --------
TOP_K = 3
MAX_CONTEXT_CHARS = 6000  # cost guard


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def main():
    load_dotenv()
    client = OpenAI()

    # 1) Validate files exist
    if not VECTORS_FILE.exists():
        raise FileNotFoundError(f"Missing: {VECTORS_FILE} (run embed_index_openai.py)")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Missing: {META_FILE} (run embed_index_openai.py)")

    # 2) Load vectors + chunks
    vectors = np.load(VECTORS_FILE).astype(np.float32)  # shape (N, D)
    chunks = json.loads(META_FILE.read_text(encoding="utf-8"))  # list[str]

    if len(chunks) == 0:
        raise RuntimeError("META_FILE has zero chunks.")
    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"Mismatch: vectors={vectors.shape[0]} but chunks={len(chunks)}. Re-run embed_index_openai.py"
        )

    # 3) Normalize stored vectors for cosine similarity
    vectors = np.array([normalize(v) for v in vectors], dtype=np.float32)

    # 4) Ask question
    q = input("Ask a question about the resume: ").strip()
    if not q:
        print("Empty question. Exiting.")
        return

    # 5) Embed the question
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
    q_vec = normalize(q_vec)

    # 6) Retrieve top chunks by cosine similarity (dot product since normalized)
    scores = vectors @ q_vec
    top_idx = np.argsort(-scores)[:TOP_K]

    # 7) Build context (with a cost guard)
    context_blocks = []
    used_idx = []
    total_chars = 0

    for i in top_idx:
        block = f"[Chunk {i} | score={scores[i]:.3f}]\n{chunks[i]}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(block)
        used_idx.append(int(i))
        total_chars += len(block)

    context_text = "\n\n".join(context_blocks)

    # 8) Ask the chat model to answer ONLY from context
    prompt = f"""
You are a strict resume analyst. Use ONLY the provided resume context.
If the answer is not in the context, say exactly: Not found in the resume.

RESUME CONTEXT:
{context_text}

QUESTION:
{q}

Rules:
- Answer in bullet points if listing items.
- Be short and specific.
- Do NOT mention sources.
"""


    resp = client.responses.create(
        model=CHAT_MODEL,
        input=prompt.strip(),
    )

    print("\n--- ANSWER ---\n")
    print(resp.output_text.strip())

    # 9) Also print which chunks were used (hard proof)
    print(f"\n(Used chunks: {used_idx})")


if __name__ == "__main__":
    main()
