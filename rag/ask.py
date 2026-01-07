from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def main():
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_DIR / "vectors.npy").astype(np.float32)

    question = input("Ask a question about your docs: ").strip()
    if not question:
        print("No question entered.")
        return

    q_resp = client.embeddings.create(model=EMBED_MODEL, input=question)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

    # cosine similarity
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    sims = vectors @ q_vec
    top_idx = np.argsort(-sims)[:5]  # top 5 chunks

    context_parts = []
    for i in top_idx:
        c = chunks[int(i)]
        context_parts.append(f"[{c['source']} chunk {c['chunk_id']} score={float(sims[int(i)]):.3f}]\n{c['text']}")

    context = "\n\n---\n\n".join(context_parts)
    print("\nUSED CONTEXT:\n")
    print(context)
    print("\n------------------\n")

    system = (
        "Answer using ONLY the context. "
        "If the context does not contain the answer, say: 'I don't know from the document.'"
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    print("\nAnswer:\n")
    print(chat.choices[0].message.content)

if __name__ == "__main__":
    main()
