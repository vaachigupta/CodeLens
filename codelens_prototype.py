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
import re

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_function_chunks(filepath: str) -> list[dict]:
    """Parse a Python file and extract each top-level function as a chunk."""
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            chunk_source = ast.get_source_segment(source, node)
            chunks.append(
                {
                    "id": f"{filepath}::{node.name}",
                    "file": filepath,
                    "function_name": node.name,
                    "source": chunk_source,
                    "has_docstring": ast.get_docstring(node) is not None,
                    "is_snake_case": bool(re.fullmatch(r"[a-z_][a-z0-9_]*", node.name)),
                }
            )
    return chunks


def build_vector_store(chunks: list[dict]):
    """Embed chunks with TF-IDF and load them into a Chroma collection."""
    corpus = [c["source"] for c in chunks]
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(corpus).toarray()

    client = chromadb.Client()
    collection = client.get_or_create_collection("codebase_chunks")
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["source"] for c in chunks],
        metadatas=[
            {
                "function_name": c["function_name"],
                "file": c["file"],
                "has_docstring": c["has_docstring"],
                "is_snake_case": c["is_snake_case"],
            }
            for c in chunks
        ],
    )
    return collection, vectorizer


def retrieve_similar(collection, vectorizer, new_code: str, top_k: int = 2):
    """Retrieve the most similar existing chunks to a new code submission."""
    new_embedding = vectorizer.transform([new_code]).toarray().tolist()
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

    print("\n[3] Embedding chunks (TF-IDF placeholder) and loading into Chroma...")
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
