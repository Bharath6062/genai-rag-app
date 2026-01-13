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

TOP_K =5
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

    # 1) Embed the question
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=q)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

    # 2) Normalize vectors for cosine similarity
    v = vectors.astype(np.float32)
    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    sims = v_norm @ q_norm
    ranked = np.argsort(-sims)

    # 3) Balanced retrieval: try to include both resume and job description
    resume_idx = []
    jd_idx = []
    for i in ranked:
        m = meta[int(i)]
        doc_id = m.get("doc_id") or m.get("source") or ""
        if "resume" in doc_id.lower():
            resume_idx.append(int(i))
        elif "job" in doc_id.lower() or "description" in doc_id.lower():
            jd_idx.append(int(i))

    # pick top_k with balance (3 + 3), fallback to overall if not enough
    k = min(6, len(meta))
    picked = []
    picked.extend(resume_idx[: max(1, k // 2)])
    picked.extend(jd_idx[: max(1, k - len(picked))])

    # fill remaining from global rank
    for i in ranked:
        i = int(i)
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)

    # 4) Build context + retrieved list
    context_parts = []
    retrieved = []
    for i in picked:
        m = meta[int(i)]
        doc_id = m.get("doc_id") or m.get("source") or "unknown"
        chunk_id = m.get("chunk_id", "unknown")
        txt = m.get("text") or texts[int(i)]
        score = float(sims[int(i)])

        context_parts.append(f"[{doc_id} chunk {chunk_id} score={score:.3f}]\n{txt}")
        retrieved.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
                "text_preview": txt[:300],
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    system = (
        "Answer using ONLY the context. "
        "If the context does not contain the answer, say: 'I don't know from the document.' "
        "If the question asks to LIST/COUNT/EXTRACT items, be complete and do not drop items. "
        "Always cite evidence like [resume.txt chunk 1]."
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    answer = chat.choices[0].message.content.strip()

    # 5) IMPORTANT: return JSON (this fixes your null)
    return {
        "question": q,
        "top_k": len(retrieved),
        "retrieved": retrieved,
        "answer": answer,
    }
