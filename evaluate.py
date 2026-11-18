"""
evaluate.py - measure how well CodeLens works, and compare it with a baseline.

For every test file listed in test_cases/manifest.json it runs:
  1. CodeLens : retriever -> prompt -> LLM -> citation check   (uses the indexed repo)
  2. Baseline : the SAME LLM, but with NO retrieved codebase context (generic review)

and scores each one:
  - Recall            : of the violations we seeded, how many did it catch?
  - False alarms      : on conformant files (which should get NO comments),
                        how many files got at least one comment?
  - Citation validity : (CodeLens only) share of comments whose citation was a
                        function that was really retrieved (not made up)

Usage:
    python evaluate.py --repo E:\\Documents\\Financial-Market-Risk-Analysis
    python evaluate.py --repo <repo> --top-k 5 --delay 3
    python evaluate.py --repo <repo> --no-baseline
    python evaluate.py --repo <repo> --mock          # offline plumbing test, meaningless scores
    python evaluate.py --rescore results\\eval_XXXX.json   # re-score a saved run, no LLM calls

IMPORTANT: "caught" is decided by keyword matching on the comment text (see
CATEGORY_RULES below). It is a quick approximation. The full comments are saved
to results/ so you can read them and double-check the scores by hand.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from chain import CodebaseRetriever, build_chain, get_llm, get_mock_llm
from codelens import build_vector_store, collect_repo_chunks


# --------------------------------------------------------------- scoring rules
def _is_param_comment(text: str) -> bool:
    return bool(re.search(r"param|argument", text))


# Each rule gets the lower-cased "issue + comment" text and says whether the
# comment is about that kind of violation.
CATEGORY_RULES = {
    "function_naming": lambda t: bool(
        re.search(r"function name|function naming|name of the function|function.{0,30}(camel|pascal|snake)|(camel|pascal)\s?case function", t)
    ) or (not _is_param_comment(t) and bool(re.search(r"camel|pascal|snake_case|naming convention|name style", t))),
    "param_naming": lambda t: _is_param_comment(t)
    and bool(re.search(r"camel|pascal|snake|lower|capital|upper|naming|case", t)),
    "vague_params": lambda t: bool(
        re.search(r"single[- ]letter|one[- ]letter|vague|non[- ]descriptive|not descriptive|cryptic|short (param|name|variable)|meaningful|descriptive (param|name|variable)|unclear|abbreviat", t)
    ),
    # strict: the comment must talk about raising AND say the codebase returns a default / does not raise
    "raises_exception": lambda t: bool(re.search(r"\brais(e|es|ed|ing)\b|valueerror", t))
    and bool(re.search(r"return(s|ing)?\b|\bnan\b|default value|instead of rais|rather than rais|(never|not|no|n't) rais", t)),
    "bare_except": lambda t: bool(
        re.search(r"bare[ `'\"]*except|except\s*:|except clause|catch[- ]all|catches all|broad except|swallow", t)
    ),
    # strict: the comment must say logging goes against the codebase, not just mention a logging detail
    "logging_module": lambda t: bool(re.search(r"logging|logger", t))
    and bool(re.search(r"not present|not used|not use|do not|does not|don't|never|none of|no log|instead of|rather than|breaks|deviat|convention|reference|existing|codebase|other (functions|modules)", t)),
    "string_format": lambda t: bool(
        re.search(r"f-string|\.format|str\.format|% ?format|%[- ]?formatting|percent|string format|format string|string interpolation", t)
    ),
    "import_star": lambda t: bool(re.search(r"import \*|star import|wildcard|from numpy import|namespace pollution", t)),
}


def comment_text(c) -> str:
    if not isinstance(c, dict):
        return str(c).lower()
    return f"{c.get('issue', '')} {c.get('comment', '')}".lower()


def categories_caught(expected: list[str], comments: list) -> list[str]:
    return [cat for cat in expected if any(CATEGORY_RULES[cat](comment_text(c)) for c in comments)]


def comments_explained(expected: list[str], comments: list) -> int:
    """How many comments match at least one expected violation."""
    return sum(1 for c in comments if any(CATEGORY_RULES[cat](comment_text(c)) for cat in expected))


# ------------------------------------------------------------------- baseline
BASELINE_SYSTEM = """You are a code reviewer. Review the code below and point out problems.
Respond with JSON only, in this shape:
{{"comments": [{{"issue": "<short label>", "comment": "<1-2 sentences>"}}]}}
"""
BASELINE_PROMPT = ChatPromptTemplate.from_messages([("system", BASELINE_SYSTEM), ("human", "{new_code}")])


def parse_baseline(raw: str) -> list:
    try:
        return list(json.loads(raw).get("comments", []))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def build_baseline_chain(llm):
    return BASELINE_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_baseline)


def get_mock_baseline_llm():
    fake = json.dumps(
        {
            "comments": [
                {"issue": "Missing docstring", "comment": "The function has no docstring."},
                {"issue": "Naming", "comment": "The function name uses camelCase."},
            ]
        }
    )
    return RunnableLambda(lambda _prompt: fake)


# ---------------------------------------------------------------------- runner
def with_retry(fn, tries: int = 4):
    """Call fn(); if Groq says 'rate limit' (429), wait and try again."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if attempt < tries and ("429" in msg or "rate" in msg):
                wait = 15 * attempt
                print(f"      (rate limited, waiting {wait}s...)")
                time.sleep(wait)
                continue
            raise


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else "n/a"


def summarize(rows: list, key: str) -> dict:
    dev = [r for r in rows if r["expected"]]
    conf = [r for r in rows if not r["expected"]]
    exp_total = sum(len(r["expected"]) for r in dev)
    caught = sum(len(categories_caught(r["expected"], r[key])) for r in dev)
    explained = sum(comments_explained(r["expected"], r[key]) for r in dev)
    dev_comments = sum(len(r[key]) for r in dev)
    conf_flagged = sum(1 for r in conf if r[key])
    conf_comments = sum(len(r[key]) for r in conf)
    return {
        "recall": pct(caught, exp_total),
        "comments_on_deviant": dev_comments,
        "comments_matching_a_seeded_violation": explained,
        "conformant_files_flagged": pct(conf_flagged, len(conf)),
        "comments_on_conformant": conf_comments,
    }


def print_report(rows: list, has_baseline: bool) -> tuple:
    """Print the results table + summaries. Returns (codelens_summary, baseline_summary, kept, dropped)."""
    cl = summarize(rows, "codelens_comments")
    bl = summarize(rows, "baseline_comments") if has_baseline else None
    kept = sum(len(r["codelens_comments"]) for r in rows)
    dropped = sum(len(r["codelens_dropped"]) for r in rows)

    print("\n" + "=" * 100)
    print(f"{'File':<18}{'Expected':<46}{'CodeLens caught':<20}{'Baseline caught'}")
    print("-" * 100)
    for r in rows:
        if r["expected"]:
            c = ",".join(categories_caught(r["expected"], r["codelens_comments"])) or "-"
            b = (",".join(categories_caught(r["expected"], r["baseline_comments"])) or "-") if has_baseline else "n/a"
        else:
            c = f"{len(r['codelens_comments'])} comment(s)"
            b = f"{len(r['baseline_comments'])} comment(s)" if has_baseline else "n/a"
        print(f"{r['file']:<18}{(','.join(r['expected']) or '(none - should be silent)')[:44]:<46}{c[:18]:<20}{b[:24]}")
    print("=" * 100)

    def show(name, s):
        print(f"\n{name}")
        print(f"  Recall of seeded violations        : {s['recall']}")
        print(f"  Conformant files wrongly flagged   : {s['conformant_files_flagged']}  ({s['comments_on_conformant']} comments in total)")
        print(f"  Comments on deviant files          : {s['comments_on_deviant']}  ({s['comments_matching_a_seeded_violation']} match a seeded violation)")

    show("CodeLens (with retrieval)", cl)
    if bl:
        show("Baseline (same LLM, no retrieval)", bl)
    print(f"\nCitation validity (CodeLens): {pct(kept, kept + dropped)} of comments cited a really-retrieved function")
    return cl, bl, kept, dropped


def save_report(rows, args, cl, bl, kept, dropped, has_baseline):
    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/eval_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"repo": args.repo, "top_k": args.top_k, "mock": args.mock, "codelens": cl, "baseline": bl,
                   "citation_validity": [kept, kept + dropped], "rows": rows}, f, indent=2)

    lines = [f"# CodeLens evaluation {stamp}", f"Repo: `{args.repo}`  |  top-k: {args.top_k}  |  mock: {args.mock}", ""]
    for r in rows:
        lines += [f"## {r['file']}", f"Expected: {', '.join(r['expected']) or '(none, should be silent)'}", "",
                  "References retrieved: " + ", ".join(f"`{x}`" for x in r["refs"]), "", "**CodeLens:**"]
        lines += [f"- [{c.get('issue')}] {c.get('comment')}  \n  cited: `{c.get('cited_source')}`" for c in r["codelens_comments"]] or ["- (no comments)"]
        if r["codelens_dropped"]:
            lines.append(f"- ({len(r['codelens_dropped'])} dropped for invalid citation)")
        if has_baseline:
            lines += ["", "**Baseline:**"]
            lines += [f"- [{c.get('issue')}] {c.get('comment')}" for c in r["baseline_comments"]] or ["- (no comments)"]
        lines.append("")
    with open(f"results/eval_{stamp}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: results/eval_{stamp}.md  (read this to double-check the scoring)  and .json")


def rescore(path: str):
    """Re-score a saved results/*.json with the CURRENT scoring rules. No LLM calls."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    has_baseline = data.get("baseline") is not None
    print(f"Re-scoring {path} (no LLM calls) ...")
    cl, bl, kept, dropped = print_report(rows, has_baseline)
    out = os.path.join("results", f"rescored_{os.path.splitext(os.path.basename(path))[0]}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": path, "codelens": cl, "baseline": bl, "citation_validity": [kept, kept + dropped]}, f, indent=2)
    print(f"\nSaved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="folder of the codebase to index")
    ap.add_argument("--cases", default="test_cases", help="folder with manifest.json + test files")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to wait between LLM calls")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--mock", action="store_true", help="fake LLM, plumbing test only")
    ap.add_argument("--rescore", metavar="JSON", help="re-score a saved results file with the current rules (no LLM calls)")
    args = ap.parse_args()

    if args.rescore:
        rescore(args.rescore)
        return
    if not args.repo:
        ap.error("--repo is required unless you use --rescore")

    with open(os.path.join(args.cases, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Indexing {args.repo} ...")
    chunks, n_files, _ = collect_repo_chunks(args.repo)
    if not chunks:
        sys.exit("No functions found - check --repo.")
    print(f"  {n_files} files, {len(chunks)} chunks")
    collection, _ = build_vector_store(chunks)

    retriever = CodebaseRetriever(collection=collection, k=min(args.top_k, len(chunks)))
    llm = get_mock_llm() if args.mock else get_llm()
    chain = build_chain(retriever, llm)
    baseline = None if args.no_baseline else build_baseline_chain(get_mock_baseline_llm() if args.mock else llm)
    has_baseline = baseline is not None

    rows = []
    for case in manifest:
        path = os.path.join(args.cases, case["file"])
        with open(path, encoding="utf-8", errors="replace") as f:
            code = f.read()
        print(f"Running {case['file']} ...")

        res = with_retry(lambda: chain.invoke({"new_code": code}))
        time.sleep(args.delay)
        base_comments = []
        if has_baseline:
            base_comments = with_retry(lambda: baseline.invoke({"new_code": code}))
            time.sleep(args.delay)

        rows.append(
            {
                "file": case["file"],
                "expected": case["violations"],
                "refs": [d.metadata["id"] for d in res["docs"]],
                "codelens_comments": res["kept"],
                "codelens_dropped": res["dropped"],
                "baseline_comments": base_comments,
            }
        )

    cl, bl, kept, dropped = print_report(rows, has_baseline)
    save_report(rows, args, cl, bl, kept, dropped, has_baseline)


if __name__ == "__main__":
    main()