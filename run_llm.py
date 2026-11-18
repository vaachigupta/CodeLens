"""
run_llm.py - CodeLens end-to-end run (LangChain chain by default).

Usage:
    python run_llm.py                              # review new_submission.py against sample_codebase/
    python run_llm.py --mock                       # offline plumbing test, no API key
    python run_llm.py path/to/file.py              # review a different file
    python run_llm.py file.py --repo path/to/repo  # index a real repo instead
    python run_llm.py file.py --repo repo --top-k 5
    python run_llm.py file.py --plain              # old direct-API path, no LangChain (for comparison)
"""

import json
import sys

from chain import CodebaseRetriever, build_chain, get_llm, get_mock_llm
from codelens import build_vector_store, collect_repo_chunks, retrieve_similar
from llm_review import call_groq, review_with_llm


def make_fake_llm(real_id: str):
    """Canned reply for the --plain path with --mock (tests parsing/validation)."""

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "comments": [
                    {"issue": "Naming convention", "comment": "Uses camelCase; the references use snake_case.", "cited_source": real_id},
                    {"issue": "Made-up citation", "comment": "This cites a function that does not exist.", "cited_source": "nowhere.py::imaginary_function"},
                ]
            }
        )

    return fake_llm


def get_option(name: str, default=None):
    """Read `--name value` from the command line."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    use_mock = "--mock" in sys.argv
    use_plain = "--plain" in sys.argv
    repo = get_option("--repo", "sample_codebase")
    top_k = int(get_option("--top-k", 3))
    # positional args = anything that is not a flag or a flag's value
    flag_values = {get_option("--repo"), get_option("--top-k")}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in flag_values]
    target = args[0] if args else "new_submission.py"

    print(f"[1] Ingesting codebase: {repo}")
    chunks, n_files, skipped = collect_repo_chunks(repo, exclude=target)
    print(f"    {n_files} file(s), {len(chunks)} function/method chunks")
    if skipped:
        print(f"    skipped {len(skipped)} file(s) that could not be parsed")
    if not chunks:
        sys.exit("    No functions found - check the --repo path.")

    print("[2] Building vector store...")
    collection, vectorizer = build_vector_store(chunks)
    top_k = min(top_k, len(chunks))

    print(f"[3] Reviewing {target} ...")
    with open(target, encoding="utf-8", errors="replace") as f:
        new_code = f.read()

    if use_plain:
        # Direct API path (no LangChain), kept for comparison.
        retrieved = retrieve_similar(collection, vectorizer, new_code, top_k=top_k)
        ref_ids = retrieved["ids"][0]
        llm_fn = make_fake_llm(ref_ids[0]) if use_mock else call_groq
        print(f"[4] Plain path: asking the LLM ({'MOCK' if use_mock else 'Groq'}) directly...")
        kept, dropped = review_with_llm(new_code, retrieved, llm_fn=llm_fn)
    else:
        retriever = CodebaseRetriever(collection=collection, k=top_k)
        llm = get_mock_llm() if use_mock else get_llm()
        chain = build_chain(retriever, llm)
        print(f"[4] LangChain: retriever | prompt | {'MOCK' if use_mock else 'ChatGroq'} | parser | validator")
        result = chain.invoke({"new_code": new_code})
        ref_ids = [d.metadata["id"] for d in result["docs"]]
        kept, dropped = result["kept"], result["dropped"]

    print("    Retrieved references:")
    for rid in ref_ids:
        print(f"      - {rid}")

    print(f"\n=== {len(kept)} validated comment(s) ===")
    for i, c in enumerate(kept, 1):
        print(f"\nComment {i} [{c['issue']}]")
        print(f"  {c['comment']}")
        print(f"  -> cited source: {c['cited_source']}")

    if dropped:
        print(f"\n({len(dropped)} comment(s) dropped: citation not in retrieved set)")


if __name__ == "__main__":
    main()