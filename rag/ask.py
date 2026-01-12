from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rag.logger import write_run_log

load_dotenv()
client = OpenAI()

# Our ingest + embed scripts write outputs here
INDEX_DIR = Path("rag/out")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def main():
    # ---- Load chunks from meta.json (supports both list and {"chunks": [...]}) ----
    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    chunks = meta["chunks"] if isinstance(meta, dict) and "chunks" in meta else meta

    # ---- Load vectors ----
    vectors = np.load(INDEX_DIR / "vectors.npy").astype(np.float32)

    # ---- Input ----
    question = input("Ask a question about your docs: ").strip()
    if not question:
        print("No question entered.")
        return

    # ---- Embed the question ----
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=question)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

    # ---- Cosine similarity ----
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    q_vec_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    sims = vectors_norm @ q_vec_norm

    # ---- Balanced retrieval: try to grab 3 JD + 3 Resume chunks ----
    k_scan = min(12, len(chunks))  # scan top candidates
    top_idx_all = np.argsort(-sims)[:k_scan]

    jd_idxs = []
    res_idxs = []

    for idx in top_idx_all:
        c = chunks[int(idx)]
        doc = c.get("doc_id", "")

        if doc == "job_description.txt" and len(jd_idxs) < 3:
            jd_idxs.append(int(idx))
        elif doc == "resume.txt" and len(res_idxs) < 3:
            res_idxs.append(int(idx))

        if len(jd_idxs) == 3 and len(res_idxs) == 3:
            break
    # Ensure we include at least one "core responsibilities" JD chunk (0-3) if available
    if len(jd_idxs) > 0:
        has_core = any(
            (chunks[i].get("doc_id") == "job_description.txt") and
            (0 <= int(chunks[i].get("chunk_id", 999)) <= 3)
            for i in jd_idxs
        )

        if not has_core:
            # Find any JD chunk with chunk_id 0-3 and swap it in
            for i in range(len(chunks)):
                c = chunks[i]
                if c.get("doc_id") == "job_description.txt":
                    try:
                        cid = int(c.get("chunk_id"))
                    except Exception:
                        continue
                    if 0 <= cid <= 3 and i not in jd_idxs:
                        jd_idxs[-1] = i  # replace one JD pick
                        break

    # If we couldn't get enough from one side, fill from remaining best indices
    combined = jd_idxs + res_idxs
    if len(combined) < min(6, len(chunks)):
        for idx in top_idx_all:
            ii = int(idx)
            if ii not in combined:
                combined.append(ii)
            if len(combined) == min(6, len(chunks)):
                break

    top_idx = np.array(combined, dtype=int)

    # ---- Build context + retrieved evidence ----
    context_parts = []
    retrieved = []

    for idx in top_idx:
        c = chunks[int(idx)]
        score = float(sims[int(idx)])

        src = c.get("doc_id", "unknown")
        cid = c.get("chunk_id", "unknown")
        txt = c.get("text", "")

        context_parts.append(f"[{src} chunk {cid} score={score:.3f}]\n{txt}")

        retrieved.append({
            "source": src,
            "chunk_id": cid,
            "score": score,
            "text_preview": txt[:300],
        })

    context = "\n\n---\n\n".join(context_parts)

    print("\nUSED CONTEXT:\n")
    print(context)
    print("\n------------------\n")

    # ---- Prompt tuned for resume vs JD comparison (no invented experience) ----
    system = (
        "You are comparing TWO documents in the CONTEXT: RESUME (doc_id=resume.txt) and JOB DESCRIPTION (doc_id=job_description.txt).\n"
        "Use ONLY the provided CONTEXT.\n"
        "Output exactly these sections:\n"
        "1) Missing TECHNICAL skills (tools/tech/engineering skills in JD but not clearly in resume)\n"
        "2) Missing DOMAIN/ROLE expectations (responsibilities in JD not evidenced in resume)\n"
        "3) Qualifications gaps (degree/years/other requirements)\n"
        "4) Matched strengths (items clearly present in both)\n"
        "5) 5 resume bullet edits\n"
        "Rules:\n"
        "- Only list a missing TECHNICAL skill if the JD names it specifically (e.g., a tool/tech). If JD is generic, do not call it missing.\n"
        "- Do not invent anything not in the context.\n"
        "- For every item, cite chunk labels like [job_description.txt chunk 2] or [resume.txt chunk 1].\n"
        "- In section 5, do NOT claim work you cannot support from RESUME context. "
        "Write bullets as either: (a) true rewrite of an existing resume bullet, or (b) clearly labeled 'Planned/Training/Exposure' items.\n"
        "- If context is insufficient, say exactly: I don't know from the document."
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    answer = chat.choices[0].message.content

    print("\nAnswer:\n")
    print(answer)

    # ---- Save log ----
    log_payload = {
        "question": question,
        "top_k": len(retrieved),
        "retrieved": retrieved,
        "answer": answer,
    }
    log_path = write_run_log("logs", log_payload)
    print(f"\n[LOG SAVED] {log_path}\n")

if __name__ == "__main__":
    main()
