from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from pathlib import Path
import json
import re
import sqlite3
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Versioning
# -----------------------------
APP_VERSION = "2.37-filter-travel"
COMPARE_PROMPT_VERSION = "compare_v37_filter_travel"


# -----------------------------
# Files
# -----------------------------
VECTORS_FILE = Path("rag/out/vectors.npy")
META_FILE = Path("rag/out/meta.json")

COMPARE_CACHE_DB = Path("rag/out/compare_cache.sqlite3")


# -----------------------------
# Models
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # kept for future use


# -----------------------------
# Retrieval defaults
# -----------------------------
TOP_K_COMPARE = 8          # number of requirements to evaluate
MAX_REQUIREMENTS = 30      # hard cap


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="GenAI RAG App", version=APP_VERSION)


# -----------------------------
# OpenAI client
# -----------------------------
load_dotenv()
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# -----------------------------
# Load index (cached)
# -----------------------------
_vectors: Optional[np.ndarray] = None
_meta: Optional[List[Dict[str, Any]]] = None


def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    global _vectors, _meta
    if _vectors is None or _meta is None:
        if not VECTORS_FILE.exists() or not META_FILE.exists():
            raise RuntimeError(
                "Index not found. Run: python rag/ingest.py to generate rag/out/vectors.npy and rag/out/meta.json"
            )
        _vectors = np.load(VECTORS_FILE)
        _meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if _vectors.shape[0] != len(_meta):
            raise RuntimeError("Index mismatch: vectors rows != meta entries")
    return _vectors, _meta


# -----------------------------
# Embeddings (batched)
# -----------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns float32 matrix of shape (len(texts), dim)
    using ONE OpenAI embeddings API call.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    c = get_client()
    res = c.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in res.data], dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def cosine_scores(matrix: np.ndarray, v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v) + 1e-12
    m_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ v) / (m_norm * v_norm)


def top_k_retrieve(query: str, k: int) -> List[Dict[str, Any]]:
    vectors, meta = load_index()
    qv = embed_text(query)
    scores = cosine_scores(vectors, qv)
    idx = np.argsort(scores)[::-1][:k]
    out = []
    for i in idx:
        m = dict(meta[int(i)])
        m["score"] = float(scores[int(i)])
        m["text_preview"] = (m.get("text") or "")[:320].replace("\r", "")
        out.append(m)
    return out


# -----------------------------
# Requirements extraction (clean, stitch fragments, dedupe)
# -----------------------------
_BAD_PREFIXES = (
    "ignite partners",
    "about the role",
    "other duties",
    "other responsibilities",
    "qualifications",
    "what you'll do",
    "what you’ll do",
)

# filter non-skill / HR / logistics lines from requirements
_BAD_CONTAINS = (
    "travel occasionally",
    "willingness to travel",
    "willing to travel",
    "evening and weekend",
    "weekend work",
    "as assigned",
    "annual summit",
    "monthly meetings",
    "meet deadlines",
    "local office",
)


def _normalize_ws(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


# Merge helpers
_FRAGMENT_PREFIXES = (
    "and ",
    "or ",
    "with ",
    "including ",
    "leveraging ",
    "by leveraging ",
    "as well as ",
    "also ",
)

_EDU_PREFIXES = (
    "bachelor",
    "master",
    "phd",
    "doctorate",
    "degree",
)

_EDU_CONTINUATIONS = (
    "or an equivalent combination",
    "or equivalent combination",
    "or equivalent",
    "related field",
    "in a related field",
    "in related field",
    "related",
)


def _is_edu_line(s: str) -> bool:
    low = s.lower().strip()
    return any(low.startswith(p) for p in _EDU_PREFIXES)


def _is_edu_continuation(s: str) -> bool:
    low = s.lower().strip()
    return any(low.startswith(p) for p in _EDU_CONTINUATIONS)


def _is_fragment_for_merging(s: str) -> bool:
    """
    Used ONLY for merging (not filtering).
    """
    s2 = s.strip()
    if not s2:
        return False
    low = s2.lower()
    if any(low.startswith(p) for p in _FRAGMENT_PREFIXES):
        return True
    # starts lowercase -> likely continuation in non-bulleted JDs
    if s2 and s2[0].islower():
        return True
    # short connector-ish junk
    if len(low) <= 35 and (low.startswith(("and", "or")) or low.endswith(("...", ","))):
        return True
    return False


def _is_fragment_prefix_only(s: str) -> bool:
    """
    Used for FILTERING: only reject obvious connector fragments.
    """
    low = s.lower().strip()
    return any(low.startswith(p) for p in _FRAGMENT_PREFIXES)


def _join_clean(a: str, b: str) -> str:
    a = a.rstrip()
    b = b.lstrip()
    if a.endswith(("-", "–", "—")):
        return f"{a} {b}"
    if a.endswith((".", ":", ")", "]")):
        return f"{a} {b}"
    return f"{a} {b}"


def _merge_requirement_fragments(items: List[str]) -> List[str]:
    """
    Merge continuation fragments into the prior line.
    Also merges multi-line education requirements into one clean line.
    """
    out: List[str] = []
    i = 0
    while i < len(items):
        cur = items[i].strip()
        if not cur:
            i += 1
            continue

        # EDUCATION MERGE
        if _is_edu_line(cur):
            merged = cur
            j = i + 1
            while j < len(items):
                nxt = items[j].strip()
                if not nxt:
                    j += 1
                    continue

                if _is_edu_line(nxt) or _is_edu_continuation(nxt):
                    merged = _join_clean(merged, nxt)
                    j += 1
                    continue

                low = nxt.lower().strip()
                if low in {"related", "related field"}:
                    merged = _join_clean(merged, nxt)
                    j += 1
                    continue

                break

            out.append(merged)
            i = j
            continue

        # FRAGMENT MERGE
        if _is_fragment_for_merging(cur) and out:
            out[-1] = _join_clean(out[-1], cur)
            i += 1
            continue

        out.append(cur)
        i += 1

    out = [re.sub(r"\s+", " ", x).strip() for x in out if x.strip()]
    return out


def _split_into_sentences_or_bullets(text: str) -> List[str]:
    """
    Returns candidate requirement-like chunks.
    - If bullets exist: use bullets.
    - Else: line-based chunks with light semicolon split, then merge fragments.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    bullets = []
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            bullets.append(ln.lstrip("-•* ").strip())
    if bullets:
        return _merge_requirement_fragments(bullets)

    parts: List[str] = []
    for ln in lines:
        for p in re.split(r";+", ln):
            p = p.strip()
            if p:
                parts.append(p)

    return _merge_requirement_fragments(parts)


def _looks_like_requirement(s: str) -> bool:
    s_str = s.strip()
    s_low = s_str.lower()

    if len(s_low) < 12:
        return False

    # DROP role-summary blobs
    if len(s_str) > 350:
        return False

    # DROP HR / logistics lines (travel, weekend, etc.)
    if any(x in s_low for x in _BAD_CONTAINS):
        return False

    # only block obvious connector-fragments (not lowercase rule)
    if _is_fragment_prefix_only(s_str):
        return False

    if any(s_low.startswith(p) for p in _BAD_PREFIXES):
        return False

    if s_low in {"qualifications", "responsibilities", "about", "as assigned"}:
        return False

    return True


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
        key = re.sub(r"\s+", " ", key)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def extract_requirements_from_job_text(job_text: str, limit: int = 14) -> List[str]:
    job_text = _normalize_ws(job_text)

    candidates = _split_into_sentences_or_bullets(job_text)
    candidates = [c for c in candidates if _looks_like_requirement(c)]
    candidates = _dedupe_keep_order(candidates)

    scored: List[Tuple[int, str]] = []
    for c in candidates:
        low = c.lower()
        score = 0
        if "experience" in low:
            score += 3
        if "sql" in low:
            score += 2
        if "azure" in low:
            score += 2
        if "etl" in low or "elt" in low:
            score += 2
        if "ability" in low or "skills" in low:
            score += 1
        if "degree" in low:
            score += 1
        scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    picked = [c for _, c in scored[:limit]]

    picked_set = set(picked)
    final = [c for c in candidates if c in picked_set]
    return final[:limit]


# -----------------------------
# Simple resume education presence check
# -----------------------------
def _resume_has_education(resume_text: str) -> bool:
    low = resume_text.lower()
    return any(x in low for x in ["education", "bachelor", "master", "b.tech", "bachelor of", "master of"])


# -----------------------------
# Keyword gates (simple + explainable)
# -----------------------------
KEYWORD_GATES = [
    {"trigger": "azure data factory", "required_keywords": ["azure data factory", "data factory", "adf"]},
    {"trigger": "sql server", "required_keywords": ["sql server"]},
    {"trigger": "dimensional", "required_keywords": ["dimensional", "star schema", "snowflake schema"]},
    {"trigger": "indexes", "required_keywords": ["index", "indexes"]},
    {"trigger": "security", "required_keywords": ["security", "privacy", "pii", "compliance"]},
]


def keyword_gate(requirement: str, resume_text: str) -> Dict[str, Any]:
    req_low = requirement.lower()
    trigger = None
    req_keys: List[str] = []
    for g in KEYWORD_GATES:
        if g["trigger"] in req_low:
            trigger = g["trigger"]
            req_keys = g["required_keywords"]
            break

    found = []
    missing = []
    if trigger:
        resume_low = resume_text.lower()
        for k in req_keys:
            if k in resume_low:
                found.append(k)
            else:
                missing.append(k)

    passes = True if not trigger else (len(found) > 0)
    return {
        "trigger": trigger,
        "required_keywords": req_keys,
        "found_keywords": found,
        "missing_keywords": missing,
        "passes": passes,
    }


def confidence_label(score: float) -> str:
    if score >= 0.55:
        return "HIGH"
    if score >= 0.42:
        return "MEDIUM"
    return "LOW"


# -----------------------------
# Compare cache
# -----------------------------
def _cache_init() -> None:
    COMPARE_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(COMPARE_CACHE_DB) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS compare_cache (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )
        con.commit()


def _cache_key(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not COMPARE_CACHE_DB.exists():
        return None
    with sqlite3.connect(COMPARE_CACHE_DB) as con:
        row = con.execute("SELECT value FROM compare_cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        return json.loads(row[0])


def cache_put(key: str, value: Dict[str, Any]) -> None:
    _cache_init()
    with sqlite3.connect(COMPARE_CACHE_DB) as con:
        con.execute(
            "INSERT OR REPLACE INTO compare_cache(key,value) VALUES(?,?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        con.commit()


# -----------------------------
# API models
# -----------------------------
class CompareRequest(BaseModel):
    question: str = "Compare resume vs job description."
    top_k: int = Field(default=TOP_K_COMPARE, ge=1, le=MAX_REQUIREMENTS)
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>GenAI RAG App</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 900px; }}
    code, pre {{ background:#f4f4f4; padding: 8px; border-radius: 8px; }}
    button {{ padding: 10px 14px; margin-right: 8px; cursor: pointer; }}
    .row {{ margin: 16px 0; }}
    .pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; background:#eee; margin-left:8px; }}
  </style>
</head>
<body>
  <h1>GenAI RAG App <span class="pill">v{APP_VERSION}</span></h1>
  <p>This app compares a Resume vs Job Description using embeddings + a simple matching rubric.</p>

  <div class="row">
    <button onclick="location.href='/docs'">Open API Docs</button>
    <button onclick="location.href='/compare'">Open Compare UI</button>
  </div>

  <h3>Quick commands</h3>
  <pre><code>curl.exe -s http://127.0.0.1:8021/health
curl.exe -s -X POST http://127.0.0.1:8021/compare -H "Content-Type: application/json" -d "{{}}"</code></pre>

</body>
</html>
"""


@app.get("/compare", response_class=HTMLResponse)
def compare_ui() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Compare</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; max-width: 980px; }
    button { padding: 10px 14px; cursor: pointer; }
    pre { background:#f4f4f4; padding: 12px; border-radius: 8px; overflow:auto; }
    input { padding: 8px; width: 120px; }
    .row { margin: 12px 0; }
    label { display:inline-block; width: 90px; }
  </style>
</head>
<body>
  <h1>Compare UI</h1>
  <div class="row">
    <label>top_k</label><input id="top_k" value="8"/>
    <label style="margin-left:20px;">threshold</label><input id="threshold" value="0.4"/>
  </div>
  <div class="row">
    <button onclick="runCompare()">Run Compare</button>
    <button onclick="location.href='/'" style="margin-left:8px;">Home</button>
  </div>
  <pre id="out">(click "Run Compare")</pre>

<script>
async function runCompare(){
  const top_k = parseInt(document.getElementById('top_k').value || '8', 10);
  const threshold = parseFloat(document.getElementById('threshold').value || '0.4');
  const payload = { top_k, threshold };

  const res = await fetch('/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const txt = await res.text();
  document.getElementById('out').textContent = txt;
}
</script>
</body>
</html>
"""


@app.get("/health")
def health() -> Dict[str, Any]:
    index_ok = VECTORS_FILE.exists() and META_FILE.exists()
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "compare_prompt_version": COMPARE_PROMPT_VERSION,
        "index_ok": bool(index_ok),
        "api_file": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd().resolve()),
    }


@app.post("/compare")
def compare(req: CompareRequest) -> JSONResponse:
    vectors, meta = load_index()

    resume_texts = [m.get("text", "") for m in meta if m.get("doc_type") == "resume"]
    job_texts = [m.get("text", "") for m in meta if m.get("doc_type") == "job"]

    if not resume_texts:
        raise HTTPException(
            status_code=400,
            detail="No resume chunks found in index. Ensure resume.txt is in data/docs and re-run ingest.",
        )
    if not job_texts:
        raise HTTPException(
            status_code=400,
            detail="No job chunks found in index. Ensure jd.txt is in data/docs and re-run ingest.",
        )

    resume_full = _normalize_ws("\n".join(resume_texts))
    job_full = _normalize_ws("\n".join(job_texts))

    req_limit = int(max(1, min(req.top_k, MAX_REQUIREMENTS)))
    requirements = extract_requirements_from_job_text(job_full, limit=req_limit)

    payload = {
        "question": req.question,
        "top_k": req.top_k,
        "threshold": req.threshold,
        "requirements": requirements,
        "compare_prompt_version": COMPARE_PROMPT_VERSION,
        "embed_model": EMBED_MODEL,
        "app_version": APP_VERSION,
    }

    key = _cache_key(payload)
    cached = cache_get(key)
    if cached:
        cached["app_version"] = APP_VERSION
        cached["compare_prompt_version"] = COMPARE_PROMPT_VERSION
        return JSONResponse(cached)

    resume_idx = [i for i, m in enumerate(meta) if m.get("doc_type") == "resume"]
    if not resume_idx:
        raise HTTPException(status_code=400, detail="No resume chunks present after filtering doc_type=resume.")

    resume_chunks = [meta[i] for i in resume_idx]
    resume_matrix = vectors[resume_idx]

    req_embeddings = embed_texts(requirements)

    matched = []
    likely_matched_needs_evidence = []
    gaps = []

    for r, rv in zip(requirements, req_embeddings):
        scores = cosine_scores(resume_matrix, rv)
        best_j = int(np.argmax(scores))
        best_score = float(scores[best_j])
        best_meta = resume_chunks[best_j]

        gate = keyword_gate(r, resume_full)

        passes_threshold = best_score >= float(req.threshold)
        passes_gate = bool(gate.get("passes", True))

        # Education override
        r_low = r.lower()
        if any(k in r_low for k in ["bachelor", "master", "degree", "phd", "doctorate"]):
            if _resume_has_education(resume_full):
                passes_threshold = True

        item = {
            "requirement": (r[:120] + "...") if len(r) > 123 else r,
            "score": round(best_score, 3),
            "confidence": confidence_label(best_score),
            "passes_threshold": passes_threshold,
            "passes_keyword_gate": passes_gate,
            "keyword_gate": gate,
            "best_resume_match": {
                "doc_id": best_meta.get("doc_id"),
                "chunk_id": best_meta.get("chunk_id"),
                "text_preview": (best_meta.get("text", "")[:260]).replace("\r", ""),
            },
        }

        if passes_threshold and passes_gate:
            matched.append(item)
        elif passes_threshold and (not passes_gate):
            likely_matched_needs_evidence.append(item)
        else:
            gaps.append(item)

    out = {
        "status": "ok",
        "question": req.question,
        "top_k": req.top_k,
        "threshold": req.threshold,
        "requirements": requirements,
        "matched": matched,
        "likely_matched_needs_evidence": likely_matched_needs_evidence,
        "gaps": gaps,
        "app_version": APP_VERSION,
        "compare_prompt_version": COMPARE_PROMPT_VERSION,
    }

    cache_put(key, out)
    return JSONResponse(out)
