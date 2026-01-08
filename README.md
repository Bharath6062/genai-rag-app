\# Resume \& Job Description RAG System



A Retrieval-Augmented Generation (RAG) app that answers natural-language questions over a resume and a job description.

It uses OpenAI embeddings + vector similarity search to retrieve relevant chunks and returns answers with source citations.



\## Tech Stack

\- Python, FastAPI

\- OpenAI API (Embeddings + Chat)

\- NumPy

\- HTML/JS frontend

\- Git



\## Project Flow (RAG)

1\. Ingest text files from `data/docs/` and split into overlapping chunks

2\. Generate embeddings for each chunk and save vectors + metadata

3\. For a user question: embed the question → similarity search → top chunks → answer grounded in retrieved context



\## Setup

```bash

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt



