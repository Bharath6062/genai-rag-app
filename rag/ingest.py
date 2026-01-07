# rag/ingest.py
# INGEST VERSION: WORD-CHUNK v3 (DEDUP FIXED)

from pathlib import Path
import re

INGEST_VERSION = "WORD-CHUNK v3"

CHUNK_SIZE_WORDS = 90
OVERLAP_WORDS = 15


# -------------------------
# Cleaning
# -------------------------
def safe_clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# -------------------------
# Chunking
# -------------------------
def chunk_by_words(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks = []

    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += step

    return chunks


# -------------------------
# File discovery (DEDUPED)
# -------------------------
def find_txt_files():
    candidates = [
        Path("data"),
        Path("data") / "docs",
    ]

    found = []
    for folder in candidates:
        if folder.exists():
            found.extend(folder.glob("*.txt"))

    # DEDUPE BY REAL ABSOLUTE PATH
    unique = []
    seen = set()
    for p in found:
        real = str(p.resolve()).lower()
        if real not in seen:
            unique.append(p)
            seen.add(real)

    return unique


# -------------------------
# Main
# -------------------------
def main():
    print(f"INGEST VERSION: {INGEST_VERSION}")

    files = find_txt_files()
    print("Files found:", [str(f) for f in files])


    if not files:
        print("❌ No .txt files found")
        return

    out_dir = Path("rag") / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        print(f"\nStep B: reading -> {path}")
        raw = path.read_text(encoding="utf-8", errors="ignore")
        print(f"Step C: read complete, chars={len(raw)}")

        cleaned = safe_clean(raw)
        print(f"Step D: cleaned chars={len(cleaned)}")

        words = cleaned.split()
        print(f"Step E0: word_count = {len(words)}")

        print("Step E1: chunking...")
        chunks = chunk_by_words(cleaned)

        print(f"Step E2: chunks created = {len(chunks)}")
        print(f"Step E3: chunk sizes = {[len(c.split()) for c in chunks]}")

        # Save chunks
        out_path = out_dir / f"{path.stem}_chunks.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                f.write(f"\n--- CHUNK {i} ---\n")
                f.write(c)

        print(f"Step F: saved -> {out_path}")


if __name__ == "__main__":
    main()
