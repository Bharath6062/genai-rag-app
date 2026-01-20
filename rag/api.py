from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Versions
# -----------------------------
APP_VERSION = "2.23-matched-gaps-gated"
COMPARE_PROMPT_VERSION = "compare_v23_matched_gaps_gated"

# -----------------------------
# Paths
# -----------------------------
VECTORS_FILE = Path("rag/out/vectors.npy")
META_FILE = Path("rag/out/meta.json")

# -----------------------------
# Models
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"

# -----------------------------
# Defaults
# -----------------------------
TOP_K_COMPARE = 8

# Stricter threshold to avoid fake matches
MATCH_THRESHOLD = 0.40

# -----------------------------
# App
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI()

_CACHE: dict[str, Any] = {"meta": None, "vectors": None, "meta_mtime": None, "vec_mtime": None}


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return None


def load_index(force: bool = False) -> tuple[list[dict[str, Any]], np.ndarray]:
    meta_mtime = _mtime(META_FILE)
    vec_mtime = _mtime(VECTORS_FILE)

    if meta_mtime is None or vec_mtime is None:
        raise HTTPException(
            status_code=500,
            detail="Index missing. Run: python rag/ingest.py (need rag/out/meta.json and rag/out/vectors.npy)"
        )

    if (
        force
        or _CACHE["meta"] is None
        or _CACHE["vectors"] is None
        or _CACHE["meta_mtime"] != meta_mtime
        or _CACHE["vec_mtime"] != vec_mtime
    ):
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        vectors = np.load(VECTORS_FILE)
        _CACHE.update({"meta": meta, "vectors": vectors, "meta_mtime": meta_mtime, "vec_mtime": vec_mtime})

    return _CACHE["meta"], _CACHE["vectors"]


def _embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=q)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _cosine_scores(vectors: np.ndarray, q: np.ndarray) -> np.ndarray:
    v_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    return v_norm @ q_norm


def retrieve(meta: list[dict[str, Any]], vectors: np.ndarray, question: str, top_k: int) -> list[dict[str, Any]]:
    qv = _embed_query(question)
    sims = _cosine_scores(vectors, qv)
    idx = np.argsort(-sims)[:top_k]

    out = []
    for i in idx:
        m = meta[int(i)]
        txt = m.get("text", "")
        out.append({
            "doc_id": m.get("doc_id"),
            "doc_type": m.get("doc_type"),
            "chunk_id": m.get("chunk_id"),
            "score": float(sims[int(i)]),
            "text_preview": (txt[:320] + "...") if len(txt) > 320 else txt,
        })
    return out


def extract_requirements(job_text: str, max_items: int = 10) -> list[str]:
    """
    Deterministic requirement extraction:
    - remove headings
    - keep lines with requirement-like keywords
    - dedupe
    """
    t = job_text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in t.split("\n"):
        line = line.strip(" \t-•")
        if line:
            lines.append(line)

    drop_exact = {
        "ignite partners",
        "about the role",
        "what you'll do",
        "what you’ll do",
        "other responsibilities",
        "qualifications",
    }

    keep_keywords = [
        "sql", "sql server", "azure", "data factory", "azure data factory",
        "ssis", "function apps", "logic apps",
        "etl", "elt", "pipeline",
        "dimensional", "relational", "model", "modeling",
        "kafka", "event hub", "sftp", "stored procedure",
        "security", "privacy", "integrity",
        "communication", "problem-solving", "problem solving",
        "test", "testing", "reliability",
        "experience", "expertise", "ability",
    ]

    reqs = []
    seen = set()

    for s in lines:
        sl = s.lower().strip(" :.-")

        if sl in drop_exact:
            continue
        if len(s) < 18:
            continue
        if not any(k in sl for k in keep_keywords):
            continue

        key = re.sub(r"\s+", " ", sl)
        if key in seen:
            continue
        seen.add(key)

        if len(s) > 220:
            s = s[:217] + "..."

        reqs.append(s)
        if len(reqs) >= max_items:
            break

    return reqs


def best_match_against_docs(
    meta: list[dict[str, Any]],
    vectors: np.ndarray,
    query: str,
    doc_type: str,
) -> dict[str, Any]:
    """
    Returns best matching chunk among only doc_type (e.g., 'resume') for the given query.
    """
    qv = _embed_query(query)
    sims = _cosine_scores(vectors, qv)

    best_i = None
    best_score = -1.0

    for i, m in enumerate(meta):
        if m.get("doc_type") != doc_type:
            continue
        score = float(sims[i])
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return {"score": None, "doc_id": None, "chunk_id": None, "text_preview": ""}

    txt = meta[best_i].get("text", "")
    return {
        "score": round(best_score, 3),
        "doc_id": meta[best_i].get("doc_id"),
        "chunk_id": meta[best_i].get("chunk_id"),
        "text_preview": (txt[:260] + "...") if len(txt) > 260 else txt,
    }


def confidence_label(score: float) -> str:
    if score >= 0.55:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def keyword_gate(requirement: str, resume_text: str) -> bool:
    """
    If a requirement contains certain keywords, demand those keywords (or close variants)
    appear in the matched resume chunk. This kills fake semantic matches.
    """
    r = requirement.lower()
    t = resume_text.lower()

    gates = [
        ("sql server", ["sql server"]),
        ("azure data factory", ["data factory", "adf"]),
        ("data factory", ["data factory", "adf"]),
        ("ssis", ["ssis"]),
        ("logic apps", ["logic app"]),
        ("function apps", ["function app"]),
        ("indexes", ["index", "indexes"]),
        ("constraints", ["constraint", "constraints"]),
        ("views", ["view", "views"]),
        ("stored procedure", ["stored procedure", "stored procedures"]),
        ("dimensional", ["dimensional", "star schema", "snowflake schema"]),
        ("relational", ["relational"]),
        ("security", ["security", "privacy", "pii", "compliance", "integrity"]),
        ("privacy", ["privacy", "pii", "compliance"]),
    ]

    for key, must_have in gates:
        if key in r:
            return any(m in t for m in must_have)

    return True


def is_non_skill_requirement(requirement: str) -> bool:
    """
    Filter out stuff we don't want to pretend is a 'matched skill'
    (degree, travel, etc.).
    """
    r = requirement.lower()
    bad_markers = [
        "bachelor", "master", "degree",
        "travel", "evening", "weekend",
        "other duties", "as assigned",
        "annual", "monthly meetings",
    ]
    return any(b in r for b in bad_markers)


class CompareRequest(BaseModel):
    question: str | None = None
    top_k: int | None = None


@app.get("/health")
def health():
    try:
        load_index()
        index_ok = True
    except Exception:
        index_ok = False

    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "compare_prompt_version": COMPARE_PROMPT_VERSION,
        "index_ok": index_ok,
        "api_file": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd().resolve()),
    }


@app.post("/compare")
def compare(req: CompareRequest):
    meta, vectors = load_index()

    question = req.question or "Compare resume vs job description."
    top_k = int(req.top_k or TOP_K_COMPARE)

    retrieved = retrieve(meta, vectors, question, top_k=top_k)

    # Build job text from retrieved job chunks (full text from meta)
    job_chunks = []
    for r in retrieved:
        if r.get("doc_type") != "job":
            continue
        for m in meta:
            if m.get("doc_id") == r["doc_id"] and m.get("chunk_id") == r["chunk_id"]:
                job_chunks.append(m.get("text", ""))
                break

    job_text = "\n".join(job_chunks)
    requirements = extract_requirements(job_text, max_items=10)

    matched = []
    gaps = []

    for req_line in requirements:
        bm = best_match_against_docs(meta, vectors, req_line, doc_type="resume")
        score = float(bm.get("score") or 0.0)
        resume_txt = bm.get("text_preview") or ""

        item = {
            "requirement": req_line,
            "score": score,
            "confidence": confidence_label(score),
            "best_resume_match": bm,
        }

        # Don't pretend degree/travel is a "matched skill"
        if is_non_skill_requirement(req_line):
            gaps.append(item)
            continue

        ok = (score >= MATCH_THRESHOLD) and keyword_gate(req_line, resume_txt)

        if ok:
            matched.append(item)
        else:
            gaps.append(item)

    sources = [f'{r["doc_id"]} chunk {r["chunk_id"]}' for r in retrieved]
    scores = [round(float(r["score"]), 3) for r in retrieved]

    return {
        "status": "ok",
        "question": question,
        "top_k": top_k,
        "requirements": requirements,
        "threshold": MATCH_THRESHOLD,
        "matched": matched,
        "gaps": gaps,
        "retrieved": retrieved,
        "sources": sources,
        "scores": scores,
        "app_version": APP_VERSION,
    }

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>GenAI RAG Compare</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; max-width: 900px; margin: auto; }
    input { width: 100%; padding: 10px; font-size: 14px; }
    button { padding: 10px 16px; font-size: 14px; margin-top: 10px; cursor: pointer; }
    pre { background: #f5f5f5; padding: 12px; overflow: auto; }
  </style>
</head>
<body>
  <h2>GenAI RAG App — Compare</h2>
  <p>This page calls <code>/compare</code> (POST) and prints JSON.</p>

  <label>Question</label>
  <input id="q" value="Compare resume vs job description."/>

  <label style="margin-top:10px; display:block;">top_k</label>
  <input id="k" value="8"/>

  <button onclick="run()">Run Compare</button>

  <h3>Result</h3>
  <pre id="out">Click “Run Compare”</pre>

<script>
async function run(){
  const q = document.getElementById("q").value;
  const k = parseInt(document.getElementById("k").value || "8", 10);

  const res = await fetch("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q, top_k: k })
  });

  const text = await res.text();
  try { document.getElementById("out").textContent = JSON.stringify(JSON.parse(text), null, 2); }
  catch(e) { document.getElementById("out").textContent = text; }
}
</script>
</body>
</html>
"""

