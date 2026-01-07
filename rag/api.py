# rag/api.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Files created earlier ----------
VECTORS_FILE = Path("rag/out/resume_vectors.npy")
META_FILE = Path("rag/out/resume_meta.json")

# ---------- Models ----------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---------- Retrieval ----------
TOP_K = 3
MAX_CONTEXT_CHARS = 6000  # cost guard


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[int]          # chunk ids used
    scores: list[float]         # similarity scores for those chunks


app = FastAPI(title="Resume RAG API", version="1.0")

# Loaded at startup
client: OpenAI | None = None
vectors: np.ndarray | None = None
chunks: list[str] | None = None
@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def startup():
    """
    Load vectors/chunks once, create OpenAI client once.
    """
    global client, vectors, chunks

    load_dotenv()
    client = OpenAI()

    if not VECTORS_FILE.exists():
        raise RuntimeError(f"Missing {VECTORS_FILE}. Run embed_index_openai.py first.")
    if not META_FILE.exists():
        raise RuntimeError(f"Missing {META_FILE}. Run embed_index_openai.py first.")

    v = np.load(VECTORS_FILE).astype(np.float32)
    c = json.loads(META_FILE.read_text(encoding="utf-8"))

    if len(c) == 0:
        raise RuntimeError("resume_meta.json has zero chunks.")
    if v.shape[0] != len(c):
        raise RuntimeError(f"Mismatch: vectors={v.shape[0]} chunks={len(c)}. Re-run embedding.")

    # Normalize vectors for cosine similarity
    v = np.array([normalize(x) for x in v], dtype=np.float32)

    vectors = v
    chunks = c


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Ask a question about the resume.
    Returns answer + sources (chunk ids) + similarity scores.
    """
    global client, vectors, chunks
    assert client is not None and vectors is not None and chunks is not None

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # Embed question
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
    q_vec = normalize(q_vec)

    # Retrieve top chunks
    scores = vectors @ q_vec
    top_idx = np.argsort(-scores)[:TOP_K]

    context_blocks = []
    used = []
    used_scores = []
    total_chars = 0

    for i in top_idx:
        block = f"[Chunk {int(i)} | score={scores[i]:.3f}]\n{chunks[int(i)]}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(block)
        used.append(int(i))
        used_scores.append(float(scores[i]))
        total_chars += len(block)

    context_text = "\n\n".join(context_blocks)

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
""".strip()

    resp = client.responses.create(model=CHAT_MODEL, input=prompt)

    return AskResponse(
        question=q,
        answer=resp.output_text.strip(),
        sources=used,
        scores=used_scores,
    )
