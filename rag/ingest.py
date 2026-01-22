from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Version / Models
# -----------------------------
INGEST_VERSION = INGEST_VERSION = "ingest_v5_no_double_overlap"
EMBED_MODEL = "text-embedding-3-small"

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data/docs")
OUT_DIR = Path("rag/out")
META_FILE = OUT_DIR / "meta.json"
VECTORS_FILE = OUT_DIR / "vectors.npy"

# -----------------------------
# Cleaning / Normalization
# -----------------------------
# Common mojibake sequences and bullet artifacts from Windows/Word copy-paste
REPL = {
    # UTF-8 <-> Windows-1252 mojibake
    "â¢": "-",     # bullet-like
    "â€¢": "-",    # bullet
    "â€“": "-",    # en-dash
    "â€”": "-",    # em-dash
    "â€˜": "'",    # left single quote
    "â€™": "'",    # right single quote
    "â€œ": '"',    # left double quote
    "â€�": '"',    # right double quote

    # Word / other bullet artifacts
    "ï·": "-",     # bullet artifact seen in your output
    "": "-",      # common Word bullet character

    # Other whitespace junk
    "\u00a0": " ", # NBSP
    "\ufffd": " ", # replacement char
}

def read_text_safely(path: Path) -> str:
    """
    Read file bytes and decode safely:
    - Try UTF-8 first
    - Fall back to cp1252 (common Windows source of mojibake)
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")

def normalize_mojibake(text: str) -> str:
    for bad, good in REPL.items():
        text = text.replace(bad, good)

    # If any stray mojibake lead bytes remain, bluntly neutralize
    text = text.replace("â", " ")
    text = text.replace("ï", " ")

    return text

def safe_clean(text: str) -> str:
    text = normalize_mojibake(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def infer_doc_type(filename: str) -> str:
    f = filename.lower()
    if "resume" in f:
        return "resume"
    if "jd" in f or "job" in f:
        return "job"
    return "doc"

def split_into_chunks(text: str, max_chars: int = 1400, overlap: int = 60) -> list[str]:
    """
    Chunk by paragraphs and pack into ~max_chars chunks.
    If a single paragraph is too large, hard-split with overlap.
    (FIXED: prevents infinite overlap loop / MemoryError.)
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
        buf = []
        buf_len = 0

    for p in paras:
        # Fits in buffer
        if buf_len + len(p) + 2 <= max_chars:
            buf.append(p)
            buf_len += len(p) + 2
            continue

        # Buffer full
        flush()

        # Hard-split huge paragraph
        if len(p) > max_chars:
            start = 0
            n = len(p)
            while start < n:
                end = min(start + max_chars, n)
                chunks.append(p[start:end].strip())

                # If reached end, stop (CRITICAL FIX)
                if end >= n:
                    break

                # Ensure forward progress even if overlap is large
                next_start = end - overlap
                if next_start <= start:
                    next_start = end
                start = next_start
        else:
            buf = [p]
            buf_len = len(p)

    flush()

    return chunks

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)

def main() -> None:
    load_dotenv()
    client = OpenAI()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise SystemExit(f"DATA_DIR not found: {DATA_DIR.resolve()}")

    files = sorted([p.name for p in DATA_DIR.glob("*.txt")])

    print(f"INGEST VERSION: {INGEST_VERSION}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Files found: {files}")

    all_meta: list[dict] = []
    all_chunks: list[str] = []
    chunk_id = 0

    for name in files:
        path = DATA_DIR / name
        doc_type = infer_doc_type(name)

        raw = read_text_safely(path)
        cleaned = safe_clean(raw)

        if not cleaned:
            print(f"WARNING: {name} empty after cleaning, skipping.")
            continue

        chunks = split_into_chunks(cleaned, max_chars=1400, overlap=120)

        for c in chunks:
            all_chunks.append(c)
            all_meta.append({
                "doc_id": name,
                "doc_type": doc_type,
                "chunk_id": chunk_id,
                "text": c,
            })
            chunk_id += 1

    if not all_chunks:
        raise SystemExit("No chunks produced. Check your data/docs/*.txt files.")

    vectors = embed_texts(client, all_chunks)

    META_FILE.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(VECTORS_FILE, vectors)

    print(f"OK: wrote {len(all_meta)} chunks -> {META_FILE}")
    print(f"OK: wrote vectors shape={vectors.shape} -> {VECTORS_FILE}")

if __name__ == "__main__":
    main()
