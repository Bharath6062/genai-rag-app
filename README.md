# GenAI RAG: Resume ↔ Job Description Analyzer

A Retrieval-Augmented Generation (RAG) application that compares a resume against a job description using local embeddings, balanced retrieval, and evidence-backed answers.

## What it does
- Ingests documents (resume + job description)
- Chunks text and builds vector embeddings
- Retrieves relevant chunks using cosine similarity
- Forces balanced retrieval across both documents
- Generates answers strictly grounded in retrieved context
- Logs evidence (source, chunk id, score) for transparency

## Project Structure
- rag/ingest.py — document ingestion and chunking
- rag/embed_index_openai.py — embedding + vector index creation
- rag/ask.py — retrieval, comparison logic, answering, logging
- demo.ps1 — one-command demo runner
- logs/ — run logs (ignored in git)
- rag/out/ — chunk metadata + vectors (ignored in git)

## How to Run
1. Add documents:
