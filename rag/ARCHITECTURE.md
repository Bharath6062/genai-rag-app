# Architecture (RAG)

## Flow
1) Ingest resume text
2) Clean + chunk into ~180-word chunks
3) Embed chunks using OpenAI embeddings
4) Save:
   - rag/out/resume_vectors.npy (vectors)
   - rag/out/resume_meta.json (chunks)
5) For a question:
   - embed question
   - cosine similarity search over vectors
   - take top-k chunks
   - send chunks to chat model to generate grounded answer
6) Serve via FastAPI:
   - POST /ask -> { answer, sources, scores }
   - GET /health -> { status }
