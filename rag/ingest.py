from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict

DATA_DIR = Path("data/docs")
OUT_DIR = Path("rag/out")
META_FILE = OUT_DIR / "meta.json"

CHUNK_WORDS = 220
OVERLAP_WORDS = 40


def fix_mojibake(t: str) -> str:
    if not t:
        return ""

    # Repair common Windows cp1252/latin1 mojibake artifacts
    if "â" in t or "ï" in t:
        try:
            t = t.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass

    return (
        t.replace("\uf0b7", "•")
        .replace("ï·", "•")
        .replace("â€¢", "•")
        .replace("â€“", "-")
        .replace("â€”", "-")
        .replace("â€™", "'")
        .replace("â€˜", "'")
        .replace("â€œ", '"')
        .replace("â€�", '"')
        .replace("âs", "'s")
        .replace("Masterâs", "Master's")
        .replace("â€", "")
    )


def basic_clean(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def word_chunk(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_words, n)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap_words)

    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No .txt files found in {DATA_DIR}. Put resume.txt and job_description.txt there.")

    meta: List[Dict] = []
    chunk_id = 0

    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        raw = fix_mojibake(raw)
        raw = basic_clean(raw)

        chunks = word_chunk(raw, CHUNK_WORDS, OVERLAP_WORDS)
        if not chunks:
            continue

        for c in chunks:
            c = fix_mojibake(c)
            meta.append(
                {
                    "doc_id": fp.name,
                    "chunk_id": chunk_id,
                    "text": c,
                }
            )
            chunk_id += 1

    if not meta:
        raise RuntimeError("No chunks created. Check input files.")

    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {len(meta)} chunks -> {META_FILE}")


if __name__ == "__main__":
    main()
