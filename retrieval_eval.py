"""Evaluate the RETRIEVAL stage on its own (no LLM, no API key, no rate limit).

Question: when we embed a new function, do the functions CodeLens retrieves belong to
the same naming family (calculate_*, get_*, forecast_*, check_*, load_*) as the new one?
Those are the neighbours whose conventions the new function should follow.

Ground truth is rule-based, not hand-picked: the family is the first word of the function
name. camelCase names are split (computeBeta -> compute) and 'compute' is treated as
'calculate' (the author's intent).

Compared against:
  * chance        - expected precision if functions were picked at random
  * tfidf         - plain bag-of-words retrieval (needs scikit-learn; skipped if missing)
  * minilm        - what CodeLens uses (MiniLM embeddings in Chroma)

Usage:
  python retrieval_eval.py --repo E:\\Documents\\Financial-Market-Risk-Analysis
  python retrieval_eval.py --repo <repo> --cases heldout_cases
"""
import argparse
import ast
import json
import os
import re
import sys
from collections import Counter

from codelens import build_vector_store, collect_repo_chunks, embed_texts

ALIASES = {"compute": "calculate", "calc": "calculate", "fetch": "get"}
KS = (1, 3, 5, 10)


def family(name: str) -> str:
    name = name.split("::")[-1].split(".")[-1]
    first = re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", name.strip("_"))[0].lower()
    return ALIASES.get(first, first)


def first_function(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def tfidf_ranker(chunks):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]+", lowercase=True)
    mat = vec.fit_transform([c["source"] for c in chunks])

    def rank(code, k):
        sims = cosine_similarity(vec.transform([code]), mat)[0]
        order = sims.argsort()[::-1][:k]
        return [chunks[i]["function_name"] for i in order]

    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cases", default="test_cases")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with open(os.path.join(args.cases, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    chunks, n_files, _ = collect_repo_chunks(args.repo)
    if not chunks:
        sys.exit("No functions found - check --repo.")
    print(f"Indexed {args.repo}: {n_files} files, {len(chunks)} functions")
    collection, _ = build_vector_store(chunks)

    fam_counts = Counter(family(c["function_name"]) for c in chunks)
    tfidf = tfidf_ranker(chunks)
    kmax = max(KS)

    methods = {"minilm": lambda code: [m["function_name"] for m in collection.query(
        query_embeddings=embed_texts([code]), n_results=min(kmax, len(chunks)))["metadatas"][0]]}
    if tfidf:
        methods["tfidf"] = lambda code: tfidf(code, kmax)

    stats = {m: {k: {"hit": 0, "prec": 0.0} for k in KS} for m in methods}
    chance = {k: 0.0 for k in KS}
    used, skipped, detail = 0, [], []
    for case in manifest:
        path = os.path.join(args.cases, case["file"])
        fn = first_function(path)
        code = open(path, encoding="utf-8", errors="replace").read()
        fam = family(fn) if fn else None
        if not fam or fam_counts[fam] == 0:
            skipped.append((case["file"], fn, fam))
            continue
        used += 1
        row = {"file": case["file"], "function": fn, "family": fam, "family_size": fam_counts[fam]}
        for m, fn_rank in methods.items():
            names = fn_rank(code)
            row[m] = names[:5]
            for k in KS:
                top = names[:k]
                good = sum(1 for n in top if family(n) == fam)
                stats[m][k]["hit"] += good > 0
                stats[m][k]["prec"] += good / max(len(top), 1)
        for k in KS:
            chance[k] += fam_counts[fam] / len(chunks)
        detail.append(row)

    if not used:
        sys.exit("No test file had a function family present in the repo.")

    lines = [f"# Retrieval-only evaluation", f"Repo: `{args.repo}` ({len(chunks)} functions) | cases: {args.cases} | files used: {used}", "",
             "Family = first word of the function name (calculate / get / forecast / check / load ...). Rule-based ground truth, no LLM.", "",
             "| Method | " + " | ".join(f"Hit@{k}" for k in KS) + " | " + " | ".join(f"Prec@{k}" for k in KS) + " |",
             "|---|" + "---|" * (2 * len(KS))]
    for m in methods:
        hits = " | ".join(f"{stats[m][k]['hit']}/{used}" for k in KS)
        precs = " | ".join(f"{100 * stats[m][k]['prec'] / used:.0f}%" for k in KS)
        lines.append(f"| {m} | {hits} | {precs} |")
    lines.append("| chance | - | - | - | - | " + " | ".join(f"{100 * chance[k] / used:.0f}%" for k in KS) + " |")
    lines += ["", "Hit@k = at least one same-family function in the top k. Prec@k = share of the top k that are same-family.", "", "## Per file (top 5, minilm)"]
    for r in detail:
        lines.append(f"- `{r['file']}` `{r['function']}` (family `{r['family']}`, {r['family_size']} in repo): " + ", ".join(r["minilm"]))
    if skipped:
        lines += ["", "Skipped (family not present in repo): " + ", ".join(f"{s[0]} ({s[1]})" for s in skipped)]
    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(args.out, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.cases))
    out = os.path.join(args.out, f"retrieval_{tag}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
