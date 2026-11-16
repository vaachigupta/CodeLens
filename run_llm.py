"""
run_llm.py - CodeLens end-to-end run with LLM-generated review comments.

Usage:
    python run_llm.py                  # real run, needs GROQ_API_KEY
    python run_llm.py --mock           # offline plumbing test, no API key
    python run_llm.py path/to/file.py  # review a different file
"""

import glob
import json
import sys

from codelens import build_vector_store, extract_function_chunks, retrieve_similar
from llm_review import call_groq, review_with_llm


def make_fake_llm(real_id: str):
    """Canned reply used ONLY to test parsing/validation without an API key.

    One comment cites a function that really was retrieved (kept), the other
    cites one that does not exist (should be dropped by the citation check).
    """

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "comments": [
                    {
                        "issue": "Naming convention",
                        "comment": "Uses camelCase; the references use snake_case.",
                        "cited_source": real_id,
                    },
                    {
                        "issue": "Made-up citation",
                        "comment": "This cites a function that does not exist.",
                        "cited_source": "nowhere.py::imaginary_function",
                    },
                ]
            }
        )

    return fake_llm


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_mock = "--mock" in sys.argv
    target = args[0] if args else "new_submission.py"

    print("[1] Ingesting codebase...")
    chunks = []
    for path in sorted(glob.glob("sample_codebase/*.py")):
        chunks.extend(extract_function_chunks(path))
    print(f"    {len(chunks)} function chunks")

    print("[2] Building vector store...")
    collection, vectorizer = build_vector_store(chunks)

    print(f"[3] Reviewing {target} ...")
    with open(target) as f:
        new_code = f.read()
    retrieved = retrieve_similar(collection, vectorizer, new_code, top_k=3)
    print("    Retrieved references:")
    for rid in retrieved["ids"][0]:
        print(f"      - {rid}")

    if use_mock:
        llm_fn = make_fake_llm(retrieved["ids"][0][0])
    else:
        llm_fn = call_groq
    print(f"[4] Asking the LLM ({'MOCK' if use_mock else 'Groq'}) for review comments...")
    kept, dropped = review_with_llm(new_code, retrieved, llm_fn=llm_fn)

    print(f"\n=== {len(kept)} validated comment(s) ===")
    for i, c in enumerate(kept, 1):
        print(f"\nComment {i} [{c['issue']}]")
        print(f"  {c['comment']}")
        print(f"  -> cited source: {c['cited_source']}")

    if dropped:
        print(f"\n({len(dropped)} comment(s) dropped: citation not in retrieved set)")


if __name__ == "__main__":
    main()
