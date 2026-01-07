from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VECTORS_FILE = Path("rag/out/vectors.npy")
META_FILE = Path("rag/out/meta.json")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

TOP_K = 3
MAX_CONTEXT_CHARS = 6000


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


class AskRequest(BaseModel):
    question: str


app = FastAPI(title="Multi-Doc RAG API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client: OpenAI | None = None
vectors: np.ndarray | None = None
meta: list[dict] | None = None
texts: list[str] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def startup():
    global client, vectors, meta, texts

    load_dotenv()
    client = OpenAI()

    if not VECTORS_FILE.exists():
        raise RuntimeError(f"Missing {VECTORS_FILE}. Run embed_index_openai.py first.")
    if not META_FILE.exists():
        raise RuntimeError(f"Missing {META_FILE}. Run ingest.py first.")

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if not meta:
        raise RuntimeError("meta.json is empty.")

    texts = [m["text"] for m in meta]

    v = np.load(VECTORS_FILE).astype(np.float32)
    if v.shape[0] != len(texts):
        raise RuntimeError(f"Mismatch: vectors={v.shape[0]} chunks={len(texts)}. Re-run embedding.")

    # Normalize for cosine similarity
    v = np.array([normalize(x) for x in v], dtype=np.float32)
    vectors = v


@app.post("/ask")
def ask(req: AskRequest):
    global client, vectors, meta, texts
    assert client is not None and vectors is not None and meta is not None and texts is not None

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # Embed question
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
    q_vec = normalize(q_vec)

    # Retrieve
    scores = vectors @ q_vec
    top_idx = np.argsort(-scores)[:TOP_K]

    context_blocks = []
    used_sources = []
    used_scores = []
    total_chars = 0

    for idx in top_idx:
        idx = int(idx)
        m = meta[idx]
        block = (
            f"[doc={m['doc_id']} chunk={m['chunk_id']} score={scores[idx]:.3f}]\n"
            f"{texts[idx]}"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        context_blocks.append(block)
        used_sources.append({"doc_id": m["doc_id"], "chunk_id": m["chunk_id"]})
        used_scores.append(float(scores[idx]))
        total_chars += len(block)

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are a strict analyst. Use ONLY the provided context.
If the answer is not in the context, say exactly: Not found in the documents.

CONTEXT:
{context_text}

QUESTION:
{q}

Rules:
- If listing items, use bullet points.
- Be short and specific.
- Do not invent facts.
""".strip()

    resp = client.responses.create(model=CHAT_MODEL, input=prompt)

    return {
        "question": q,
        "answer": resp.output_text.strip(),
        "sources": used_sources,
        "scores": used_scores,
    }
