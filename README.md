# CodeLens - Milestone 1 Prototype

Explainable Retrieval-Augmented Agent for Codebase-Aware Code Review.

## What this does
1. Parses functions out of an existing codebase (sample_codebase/)
2. Embeds each function (TF-IDF placeholder for now)
3. Stores embeddings in a Chroma vector database
4. Given a new code submission (new_submission.py), retrieves the most similar
   existing function
5. Generates review comments (rule-based placeholder for now) that cite the
   exact existing function each comment is based on

## Not yet implemented (next milestone)
- Real semantic embeddings (transformer-based) instead of TF-IDF
- LLM-based review comment generation instead of rule-based checks
- LangChain agent orchestration
- Web interface (FastAPI + Streamlit/React)

## Run it
```
pip install -r requirements.txt
python codelens.py
```
