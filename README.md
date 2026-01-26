# Resume–Job Description Matcher (GenAI + Rules)

## Problem
Candidates don’t know why their resume fails ATS screening.

## Solution
A deterministic tool that compares a resume with a job description using:
- semantic embeddings
- keyword evidence checks
- clean requirement extraction
- cache-safe comparisons

## Features
- Extracts clean requirements from messy JDs
- Matches each requirement to resume evidence
- Uses explainable keyword gates (SQL, Azure, Security, etc.)
- Prevents stale results with content-hash caching
- Shows clear matched / missing gaps

## How it works
1. Ingest resume + JD
2. Create embeddings
3. Store vectors
4. Compare requirements vs resume chunks
5. Apply keyword gates
6. Cache results safely

## How to run
python rag/ingest.py  
uvicorn rag.api:app --reload  
POST /compare

## Why this matters
This shows real GenAI engineering: reliability, explainability, and reproducibility.
