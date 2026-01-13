from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VECTORS_FILE = Path("rag/out/vectors.npy")
META_FILE = Path("rag/out/meta.json")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

MAX_CONTEXT_CHARS = 6000


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


class AskRequest(BaseModel):
    question: str


app = FastAPI(title="Multi-Doc RAG API", version="2.4")
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

    # Normalize once for cosine similarity
    v = np.array([normalize(x) for x in v], dtype=np.float32)
    vectors = v


def clean_text(t: str) -> str:
    if not t:
        return ""
    return (
        t.replace("\uf0b7", "•")
        .replace("â€™", "'")
.replace("â€˜", "'")
.replace("â€œ", '"')
.replace("â€�", '"')
.replace("â€“", "-")
.replace("â€”", "-")
.replace("â€¢", "•")
.replace("â€", "")
.replace("âs", "'s")
.replace("Masterâs", "Master's")
    )


def _is_resume_cite(cite: str) -> bool:
    return "resume" in cite.lower()


def _is_jd_cite(cite: str) -> bool:
    c = cite.lower()
    return ("job" in c) or ("description" in c)


def _extract_first_json_object(text: str) -> dict | None:
    """
    Extract first JSON object from text. Handles cases where model adds extra text.
    """
    if not text:
        return None

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _validate_compare_payload(payload: dict, allowed_set: set[str]) -> tuple[bool, str]:
    """
    payload format:
    {
      "matched": [{"claim": "...", "jd_cite": "[job_description.txt chunk 6]", "resume_cite": "[resume.txt chunk 0]"}],
      "gaps":    [{"claim": "...", "jd_cite": "...", "resume_cite": "..."}],
      "upgrades":[{"claim": "...", "jd_cite": "...", "resume_cite": "..."}]
    }
    """
    for key in ["matched", "gaps", "upgrades"]:
        if key not in payload or not isinstance(payload[key], list):
            return False, f"Missing or invalid '{key}' list."

        for item in payload[key]:
            if not isinstance(item, dict):
                return False, f"Invalid item type in '{key}'."

            claim = item.get("claim")
            jd_cite = (item.get("jd_cite") or "").strip()
            resume_cite = (item.get("resume_cite") or "").strip()

            # write back stripped cites so rendering is clean
            item["jd_cite"] = jd_cite
            item["resume_cite"] = resume_cite

            if not isinstance(claim, str) or not claim.strip():
                return False, f"Empty claim in '{key}'."

            if jd_cite not in allowed_set:
                return False, f"Invalid jd_cite in '{key}': {jd_cite}"

            if resume_cite not in allowed_set:
                return False, f"Invalid resume_cite in '{key}': {resume_cite}"

            # Ensure types match: jd_cite must be JD, resume_cite must be resume
            if not _is_jd_cite(jd_cite):
                return False, f"jd_cite is not a JD citation: {jd_cite}"
            if not _is_resume_cite(resume_cite):
                return False, f"resume_cite is not a resume citation: {resume_cite}"

    return True, "ok"


def _render_compare_text(payload: dict) -> str:
    """
    Convert validated compare JSON into readable text.
    """
    def fmt(items: list[dict]) -> str:
        if not items:
            return "- (none)"
        lines = []
        for it in items:
            claim = it["claim"].strip()
            lines.append(f"- {claim} {it['jd_cite']} {it['resume_cite']}")
        return "\n".join(lines)

    out = []
    out.append("Matched:")
    out.append(fmt(payload.get("matched", [])))
    out.append("\nGaps:")
    out.append(fmt(payload.get("gaps", [])))
    out.append("\nResume upgrades:")
    out.append(fmt(payload.get("upgrades", [])))
    return "\n".join(out)


@app.post("/ask")
def ask(req: AskRequest):
    global client, vectors, meta, texts
    assert client is not None and vectors is not None and meta is not None and texts is not None

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    q_low = q.lower()

    # intent routing
    only_resume = ("resume" in q_low) and (
        "my resume" in q_low or "in my resume" in q_low or "from my resume" in q_low
    )
    only_jd = ("job description" in q_low) or (" jd " in f" {q_low} ") or q_low.startswith("jd ")
    is_compare = any(w in q_low for w in ["compare", "gap", "gaps", "missing", "match", "alignment", "fit"])

    # retrieval size
    k_normal = 6
    k_compare = 10
    k = min(k_compare, len(meta)) if is_compare else min(k_normal, len(meta))

    # 1) Embed question
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=q)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

    # 2) Cosine similarity (vectors normalized already)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    sims = vectors @ q_norm
    ranked = np.argsort(-sims)

    # 3) Build resume/jd pools
    resume_idx: list[int] = []
    jd_idx: list[int] = []
    for i in ranked:
        m = meta[int(i)]
        doc_id = (m.get("doc_id") or m.get("source") or "").lower()
        if "resume" in doc_id:
            resume_idx.append(int(i))
        elif "job" in doc_id or "description" in doc_id:
            jd_idx.append(int(i))

    # 3.1) For compare mode, force-include an education chunk from resume if we can find it
    edu_pick: int | None = None
    if is_compare and resume_idx:
        for i in resume_idx[:30]:
            t = (meta[int(i)].get("text") or texts[int(i)] or "").lower()
            if "education" in t or "master" in t or "bachelor" in t:
                edu_pick = int(i)
                break

    # 4) Pick chunks based on intent
    if is_compare:
        picked: list[int] = []
        picked.extend(resume_idx[: max(1, k // 2)])
        picked.extend(jd_idx[: max(1, k - len(picked))])

        if edu_pick is not None and edu_pick not in picked and len(picked) < k:
            picked.append(edu_pick)

        for i in ranked:
            i = int(i)
            if len(picked) >= k:
                break
            if i not in picked:
                picked.append(i)

    elif only_resume:
        picked = resume_idx[:k]

    elif only_jd:
        picked = jd_idx[:k]

    else:
        picked = []
        picked.extend(resume_idx[: max(1, k // 2)])
        picked.extend(jd_idx[: max(1, k - len(picked))])
        for i in ranked:
            i = int(i)
            if len(picked) >= k:
                break
            if i not in picked:
                picked.append(i)

    if not picked:
        picked = [int(i) for i in ranked[:k]]

    # 5) Build context + retrieved list with MAX_CONTEXT_CHARS
    context_parts: list[str] = []
    retrieved: list[dict] = []
    total_chars = 0

    for i in picked:
        m = meta[int(i)]
        doc_id = m.get("doc_id") or m.get("source") or "unknown"
        chunk_id = m.get("chunk_id", "unknown")
        txt = m.get("text") or texts[int(i)]
        txt = clean_text(txt)
        score = float(sims[int(i)])

        block = f"[{doc_id} chunk {chunk_id} score={score:.3f}]\n{txt}\n"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(block)
        total_chars += len(block)

        retrieved.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
                "text_preview": txt[:300],
            }
        )

    sources = [f"{r['doc_id']} chunk {r['chunk_id']}" for r in retrieved]
    scores = [round(float(r["score"]), 3) for r in retrieved]

    if not retrieved:
        return {
            "question": q,
            "top_k": k,
            "answer": "I don't know from the document.",
            "retrieved": [],
            "sources": [],
            "scores": [],
        }

    context = "\n\n---\n\n".join(context_parts)

    # Allowed citation tokens the model may use (exact)
    allowed_list = [f"[{r['doc_id']} chunk {r['chunk_id']}]" for r in retrieved]
    allowed_set = set(allowed_list)

    # ============================
    # Compare mode: JSON + validate
    # ============================
    if is_compare:
        system = (
            "You are comparing TWO documents: a resume and a job description. "
            "Use ONLY the provided context.\n\n"
            "Return ONLY valid JSON. No markdown. No extra text.\n\n"
            "JSON schema (must follow exactly):\n"
            "{\n"
            '  "matched": [{"claim": "...", "jd_cite": "[...]", "resume_cite": "[...]"}],\n'
            '  "gaps":    [{"claim": "...", "jd_cite": "[...]", "resume_cite": "[...]"}],\n'
            '  "upgrades":[{"claim": "...", "jd_cite": "[...]", "resume_cite": "[...]"}]\n'
            "}\n\n"
            "Rules:\n"
            "- For each gap/upgrade, choose the MOST relevant resume chunk among Allowed Citations (prefer experience/skills chunks over summary if they exist).\n"

            "- If any Allowed Citations include [job_description.txt chunk 5], you MUST include at least 1 gap about security/compliance unless the resume context explicitly mentions it.\n"
            "- If any Allowed Citations include [job_description.txt chunk 4], you MUST include at least 1 gap about design docs / design hypotheses unless the resume context explicitly mentions it.\n"

"- Each item MUST have jd_cite from the job description and resume_cite from the resume.\n"
"- jd_cite and resume_cite MUST be EXACTLY one of the Allowed Citations.\n"
"- Do NOT invent missing info.\n"
"- IMPORTANT: If the resume evidence supports a JD requirement, it MUST go under 'matched', not 'gaps'.\n"
"- Degree logic: If resume shows a Master's degree but the major/field is not visible in retrieved text, it is NOT a gap.\n"
"  Put it in 'matched' as 'Master’s degree present (field not shown in retrieved text)'.\n"
"  Then add an 'upgrade' telling to specify the major/field.\n"
"- Analytics logic: If resume mentions business analysis/reporting/analytics work, do NOT call 'business analytics' a gap.\n"
"- Output at least 2 'upgrades' if possible.\n"
"- If truly unsure, leave the item out.\n\n"

            + "\n".join([f"- {c}" for c in allowed_list])
        )

        user = f"QUESTION:\n{q}\n\nCONTEXT:\n{context}"

        last_raw = ""
        for attempt in range(2):
            chat = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            last_raw = (chat.choices[0].message.content or "").strip()

            payload = _extract_first_json_object(last_raw)
            if payload is None:
                print("COMPARE_JSON_VALIDATION:", False, "No JSON object found")
                print("COMPARE_RAW_MODEL_OUTPUT:", last_raw[:2000])
                user = user + "\n\nREMINDER: Output ONLY JSON. No extra text."
                continue

            ok, reason = _validate_compare_payload(payload, allowed_set)
            print("COMPARE_JSON_VALIDATION:", ok, reason)
            if not ok:
                print("COMPARE_RAW_MODEL_OUTPUT:", last_raw[:2000])

            if ok:
                answer_text = _render_compare_text(payload)
                return {
                    "question": q,
                    "top_k": k,
                    "answer": answer_text,
                    "retrieved": retrieved,
                    "sources": sources,
                    "scores": scores,
                }

            user = user + f"\n\nYour last JSON was invalid because: {reason}\nFix it. Output ONLY JSON."

        return {
            "question": q,
            "top_k": k,
            "answer": "I don't know from the document.",
            "retrieved": retrieved,
            "sources": sources,
            "scores": scores,
        }

    # ============================
    # Normal Q&A mode: strict cites
    # ============================
    allowed_citations = "\n".join([f"- {c}" for c in allowed_list])

    system = (
        "Answer using ONLY the context. "
        "If the context does not contain the answer, say: 'I don't know from the document.' "
        "If the question asks to LIST/COUNT/EXTRACT items, be complete and do not drop items.\n\n"
        "CITATIONS RULE (MANDATORY): You MUST cite using ONLY the exact tokens from Allowed Citations.\n"
        "Allowed Citations:\n"
        f"{allowed_citations}\n"
    )

    user = f"QUESTION:\n{q}\n\nCONTEXT:\n{context}\n\nReminder: use ONLY Allowed Citations."

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    answer = (chat.choices[0].message.content or "").strip()

    return {
        "question": q,
        "top_k": k,
        "answer": answer,
        "retrieved": retrieved,
        "sources": sources,
        "scores": scores,
    }
