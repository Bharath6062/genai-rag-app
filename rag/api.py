from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Files
# -----------------------------
VECTORS_FILE = Path("rag/out/vectors.npy")
META_FILE = Path("rag/out/meta.json")

# These docs are created by ingest.py from your project-level folder: data/docs
# (Ingest writes meta.json, embed writes vectors.npy)
# data/docs/resume.txt
# data/docs/job_description.txt

# -----------------------------
# Models
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------
# Retrieval defaults
# -----------------------------
TOP_K_ASK = 6
TOP_K_COMPARE = 10
MAX_CONTEXT_CHARS = 12000  # give compare enough room


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def _doc_type(doc_id: str) -> str:
    s = (doc_id or "").lower()
    if "resume" in s:
        return "resume"
    if "job" in s or "description" in s:
        return "job"
    return "other"


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class AskRequest(BaseModel):
    question: str


app = FastAPI(title="Multi-Doc RAG API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client: OpenAI | None = None
vectors: np.ndarray | None = None
meta: list[dict[str, Any]] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def startup():
    """
    Loads vectors + meta once at boot.
    """
    global client, vectors, meta

    load_dotenv()
    client = OpenAI()

    if not META_FILE.exists():
        raise RuntimeError(f"Missing {META_FILE}. Run: python rag\\ingest.py")
    if not VECTORS_FILE.exists():
        raise RuntimeError(f"Missing {VECTORS_FILE}. Run: python rag\\embed_index_openai.py")

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if not meta:
        raise RuntimeError("meta.json is empty.")

    v = np.load(VECTORS_FILE).astype(np.float32)
    if v.shape[0] != len(meta):
        raise RuntimeError(f"Mismatch: vectors={v.shape[0]} chunks={len(meta)}. Re-run ingest + embed.")

    vectors = _normalize_rows(v)


def _embed_query(q: str) -> np.ndarray:
    assert client is not None
    resp = client.embeddings.create(model=EMBED_MODEL, input=q)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return _normalize_vec(v)


def _rank_chunks(q_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      sims: cosine similarities (N,)
      ranked: indices sorted by descending similarity
    """
    assert vectors is not None
    sims = vectors @ q_vec
    ranked = np.argsort(-sims)
    return sims, ranked


def _pick_balanced(meta_list: list[dict[str, Any]], ranked: np.ndarray, k: int) -> list[int]:
    """
    Balanced selection: try to include resume + job chunks.
    """
    resume_idx: list[int] = []
    job_idx: list[int] = []
    other_idx: list[int] = []

    for i in ranked:
        m = meta_list[int(i)]
        doc_id = str(m.get("doc_id") or m.get("source") or "unknown")
        t = _doc_type(doc_id)
        if t == "resume":
            resume_idx.append(int(i))
        elif t == "job":
            job_idx.append(int(i))
        else:
            other_idx.append(int(i))

    picked: list[int] = []

    # Aim roughly half+half for compare; for ask we still allow mix but prioritize relevant.
    half = max(1, k // 2)
    picked.extend(resume_idx[:half])
    picked.extend(job_idx[: (k - len(picked))])

    # Fill remainder from global rank
    for i in ranked:
        ii = int(i)
        if len(picked) >= k:
            break
        if ii not in picked:
            picked.append(ii)

    return picked[:k]


def _build_context(picked: list[int], sims: np.ndarray) -> tuple[str, list[dict[str, Any]], list[str], list[float]]:
    """
    Returns:
      context_text
      retrieved (debug)
      sources list
      scores list
    """
    assert meta is not None

    retrieved: list[dict[str, Any]] = []
    parts: list[str] = []

    for idx in picked:
        m = meta[idx]
        doc_id = str(m.get("doc_id") or m.get("source") or "unknown")
        chunk_id = m.get("chunk_id", idx)
        txt = str(m.get("text") or "")
        score = float(sims[idx])

        # clean text a bit (helps model + frontend)
        txt_clean = txt.strip()

        parts.append(f"[{doc_id} chunk {chunk_id}]\n{txt_clean}")

        retrieved.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
                "text_preview": _compact_whitespace(txt_clean)[:300],
            }
        )

    context = "\n\n---\n\n".join(parts)

    # clip context to avoid huge prompts
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[TRUNCATED]"

    sources = [f"{r['doc_id']} chunk {r['chunk_id']}" for r in retrieved]
    scores = [round(float(r["score"]), 3) for r in retrieved]
    return context, retrieved, sources, scores


def _chat_answer(system: str, user: str) -> str:
    assert client is not None
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,  # IMPORTANT: deterministic output
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


@app.post("/ask")
def ask(req: AskRequest):
    """
    General Q&A over all indexed docs.
    Returns JSON including retrieved chunks + sources + scores.
    """
    assert meta is not None

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    q_vec = _embed_query(q)
    sims, ranked = _rank_chunks(q_vec)

    k = min(TOP_K_ASK, len(meta))
    picked = _pick_balanced(meta, ranked, k)
    context, retrieved, sources, scores = _build_context(picked, sims)

    system = (
        "You are a strict RAG assistant.\n"
        "Rules:\n"
        "1) Answer using ONLY the provided context.\n"
        "2) If the context does not contain the answer, reply exactly: I don't know from the document.\n"
        "3) Always cite like [resume.txt chunk 1] (use the bracket headers from context).\n"
        "4) If asked to list items, be complete and do not invent items.\n"
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"
    answer = _chat_answer(system, user)

    return {
        "question": q,
        "top_k": k,
        "answer": answer,
        "retrieved": retrieved,
        "sources": sources,
        "scores": scores,
    }


@app.post("/compare")
def compare():
    """
    Fixed compare endpoint: compares resume vs job description and lists:
    - Matched (resume evidence + job evidence)
    - Gaps (job requirement evidence + 'not found in resume' statement)
    - Resume upgrades (actionable bullet changes)
    """
    assert meta is not None

    q = "Compare my resume vs the job description and list gaps"

    q_vec = _embed_query(q)
    sims, ranked = _rank_chunks(q_vec)

    k = min(TOP_K_COMPARE, len(meta))
    picked = _pick_balanced(meta, ranked, k)
    context, retrieved, sources, scores = _build_context(picked, sims)

    system = (
        "You are a strict resume-vs-job comparator.\n"
        "Rules:\n"
        "1) Use ONLY the provided context.\n"
        "2) Output EXACTLY three sections with headings:\n"
        "   Matched:\n"
        "   Gaps:\n"
        "   Resume upgrades:\n"
        "3) Every bullet MUST include at least one citation like [resume.txt chunk 2] or [job_description.txt chunk 1].\n"
        "4) For a 'Gap' bullet: cite the job requirement, and only call it a gap if you cannot find supporting resume evidence in context.\n"
        "5) If the context is insufficient to compare, say: I don't know from the document.\n"
        "6) No fluff. No invented claims.\n"
    )

    user = f"CONTEXT:\n{context}\n\nTASK:\n{q}"
    answer = _chat_answer(system, user)

    return {
        "question": q,
        "top_k": k,
        "answer": answer,
        "retrieved": retrieved,
        "sources": sources,
        "scores": scores,
    }
