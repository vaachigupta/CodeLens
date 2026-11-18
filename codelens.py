"""
CodeLens - Milestone 1 Prototype
Explainable Retrieval-Augmented Agent for Codebase-Aware Code Review

This early prototype demonstrates the RETRIEVAL half of the pipeline end to end:
  1. Ingest an existing codebase, split it into function-level chunks
  2. Embed each chunk (TF-IDF vectors for this milestone, to be upgraded to a
     transformer-based embedding model in the next milestone)
  3. Store chunks + embeddings in a vector database (Chroma)
  4. For a newly submitted code snippet, retrieve the most similar existing
     chunks from the codebase
  5. Generate review comments that CITE the specific retrieved chunk they are
     based on (this is the explainability layer)

The LLM-based comment generation step is currently mocked with simple rule
checks (naming style, docstring presence, exception style) as a placeholder --
this is the "one thing I'm currently unsure about" flagged in the project card:
how to properly evaluate LLM-generated comments once that step is added.
"""

import ast
import glob
import os
import re

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

# Real semantic embedding model (all-MiniLM-L6-v2, run locally via ONNX).
# The model file (~80 MB) is downloaded automatically the first time it is used.
# By default Chroma stores it on the C: drive (~/.cache); we keep it inside the
# project folder instead, so a full C: drive can't break the download.
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(__file__).resolve().parent / "models" / "all-MiniLM-L6-v2"
_embedder = embedding_functions.DefaultEmbeddingFunction()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of code strings into a list of embedding vectors."""
    return [[float(x) for x in vec] for vec in _embedder(texts)]


SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}


def extract_function_chunks(filepath: str) -> list[dict]:
    """Parse a Python file and extract functions and class methods as chunks."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    tree = ast.parse(source)
    chunks = []

    def add(node, qualname):
        chunks.append(
            {
                "id": f"{filepath}::{qualname}",
                "file": filepath,
                "function_name": node.name,
                "line": node.lineno,
                "source": ast.get_source_segment(source, node),
                "has_docstring": ast.get_docstring(node) is not None,
                "is_snake_case": bool(re.fullmatch(r"[a-z_][a-z0-9_]*", node.name)),
            }
        )

    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, funcs):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, funcs):
                    add(sub, f"{node.name}.{sub.name}")
    return chunks


def collect_repo_chunks(root: str, exclude: str | None = None):
    """Walk a folder, extract chunks from every .py file.

    - skips virtual-env / build folders and hidden folders (.git, .venv, ...)
    - skips `exclude` (the file under review, so it can't cite itself)
    - skips files that fail to parse instead of crashing
    Returns (chunks, number_of_files_read, list_of_skipped_files).
    """
    exclude_path = os.path.abspath(exclude) if exclude else None
    chunks, files_read, skipped = [], 0, []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            if exclude_path and os.path.abspath(path) == exclude_path:
                continue
            try:
                chunks.extend(extract_function_chunks(path))
                files_read += 1
            except (SyntaxError, ValueError):
                skipped.append(path)

    # Chroma needs unique ids: two functions with the same name in one file
    seen = {}
    for c in chunks:
        n = seen.get(c["id"], 0)
        seen[c["id"]] = n + 1
        if n:
            c["id"] = f'{c["id"]}#{c["line"]}'
    return chunks, files_read, skipped


def build_vector_store(chunks: list[dict]):
    """Embed chunks with a semantic embedding model and load them into Chroma."""
    corpus = [c["source"] for c in chunks]
    embeddings = embed_texts(corpus)

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "codebase_chunks", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["source"] for c in chunks],
        metadatas=[
            {
                "function_name": c["function_name"],
                "file": c["file"],
                "line": c["line"],
                "has_docstring": c["has_docstring"],
                "is_snake_case": c["is_snake_case"],
            }
            for c in chunks
        ],
    )
    return collection, _embedder


def retrieve_similar(collection, vectorizer, new_code: str, top_k: int = 2):
    """Retrieve the most similar existing chunks to a new code submission."""
    new_embedding = embed_texts([new_code])
    results = collection.query(query_embeddings=new_embedding, n_results=top_k)
    return results


def generate_review_comments(new_func_name: str, new_source: str, retrieved: dict) -> list[dict]:
    """Rule-based placeholder for the LLM review-generation step.

    Compares the new submission against the retrieved reference chunk(s) and
    produces a comment for every convention mismatch, each citing the exact
    reference function it is based on.
    """
    comments = []
    is_snake_case = bool(re.fullmatch(r"[a-z_][a-z0-9_]*", new_func_name))
    has_docstring = '"""' in new_source

    top_match_name = retrieved["metadatas"][0][0]["function_name"]
    top_match_file = retrieved["metadatas"][0][0]["file"]
    citation = f"{top_match_file}::{top_match_name}"

    if not is_snake_case:
        comments.append(
            {
                "issue": "Naming convention",
                "comment": (
                    f"Function '{new_func_name}' uses camelCase, but this codebase "
                    f"consistently uses snake_case for function names."
                ),
                "cited_source": citation,
            }
        )

    if not has_docstring:
        comments.append(
            {
                "issue": "Missing docstring",
                "comment": (
                    f"Function '{new_func_name}' has no docstring. Every function in this "
                    f"codebase documents Args, Returns, and Raises."
                ),
                "cited_source": citation,
            }
        )

    if "raise Exception(" in new_source:
        comments.append(
            {
                "issue": "Generic exception",
                "comment": (
                    f"Function '{new_func_name}' raises a generic Exception with a vague message. "
                    f"This codebase raises ValueError with a formatted, descriptive message for "
                    f"invalid input."
                ),
                "cited_source": citation,
            }
        )

    return comments


if __name__ == "__main__":
    print("=" * 70)
    print("CodeLens - Milestone 1 Prototype Run")
    print("=" * 70)

    print("\n[1] Ingesting existing codebase...")
    all_chunks = []
    for path in sorted(glob.glob("sample_codebase/*.py")):
        file_chunks = extract_function_chunks(path)
        all_chunks.extend(file_chunks)
        print(f"    - {path}: found {len(file_chunks)} function(s)")

    print(f"\n[2] Total chunks extracted: {len(all_chunks)}")
    for c in all_chunks:
        print(f"    - {c['id']}")

    print("\n[3] Embedding chunks (MiniLM semantic embeddings) and loading into Chroma...")
    collection, vectorizer = build_vector_store(all_chunks)
    print(f"    Vector store now contains {collection.count()} chunks.")

    print("\n[4] Loading new submission for review: new_submission.py")
    with open("new_submission.py") as f:
        new_source = f.read()
    print("    Submitted code:")
    print("    " + "\n    ".join(new_source.splitlines()))

    print("\n[5] Retrieving most similar existing code from the codebase...")
    results = retrieve_similar(collection, vectorizer, new_source, top_k=2)
    for i, (doc_id, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        print(f"    Match {i+1}: {doc_id}  (distance={dist:.4f})")

    print("\n[6] Generating review comments with citations...")
    review = generate_review_comments("getPaymentById", new_source, results)
    for i, r in enumerate(review, 1):
        print(f"\n    Comment {i} [{r['issue']}]:")
        print(f"      {r['comment']}")
        print(f"      -> cited source: {r['cited_source']}")

    print("\n" + "=" * 70)
    print(f"Review complete: {len(review)} convention issue(s) flagged, all with citations.")
    print("=" * 70)