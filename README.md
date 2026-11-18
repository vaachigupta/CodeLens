# CodeLens

**An Explainable Retrieval-Augmented Agent for Codebase-Aware Code Review**

Generic linters know nothing about a project's own conventions and cannot justify
their suggestions. CodeLens indexes an existing Python codebase, retrieves the
functions most similar to a new piece of code, and asks an LLM to review the new
code against what that codebase actually does. Every review comment must cite the
exact existing function it is based on, and any comment whose citation was not
really retrieved is thrown away.

GenAI course project (Track B, applied syllabus). Milestone 1 submission files
(`GenAI_milestone1.docx` / `.pdf`) are kept in this repo.

## How it works

```
new code
   |
   v
[Retriever]   most similar functions from the indexed codebase (Chroma, MiniLM embeddings)
   |
   v
[Prompt]      references + new code, with rules for what counts as a deviation
   |
   v
[LLM]         ChatGroq (openai/gpt-oss-120b) returns review comments as JSON
   |
   v
[Parser]      model reply -> plain text
   |
   v
[Validator]   keep only comments whose cited function was really retrieved
```

The whole thing is one LangChain chain written with the `|` (LCEL) syntax.

1. **Chunking** (`codelens.py`): Python's `ast` module splits every `.py` file into
   function and class-method chunks. Virtual-env, build and hidden folders are
   skipped, files that fail to parse are skipped, and the file under review is
   excluded so it cannot cite itself.
2. **Embedding**: each chunk is embedded with `all-MiniLM-L6-v2`, run locally through
   ONNX via Chroma. The model files are stored in `models/` inside the project
   (downloaded automatically on first run, about 80 MB, not committed to git).
3. **Vector store**: Chroma collection with cosine distance.
4. **Retrieval** (`chain.py`): `CodebaseRetriever`, a LangChain `BaseRetriever` over the
   Chroma collection. Each function becomes a LangChain `Document`.
5. **Review** (`chain.py`, `llm_review.py`): a `ChatPromptTemplate` feeds the references and
   the new code to ChatGroq, which replies in JSON.
6. **Citation check** (`llm_review.py`): comments citing anything outside the retrieved
   set are dropped, which stops the model from inventing sources.

## Files

| File | Purpose |
|---|---|
| `codelens.py` | Chunking, embeddings, Chroma vector store, retrieval |
| `llm_review.py` | System prompt, Groq call, JSON parsing, citation validation |
| `chain.py` | LangChain chain: retriever, prompt, model, parser, validator (plus an offline mock model) |
| `run_llm.py` | Command line runner for reviewing one file |
| `evaluate.py` | Evaluation harness: CodeLens versus a no-retrieval baseline |
| `test_cases/` | 15 seeded test files and `manifest.json` (what each file contains) |
| `heldout_cases/` | 8 separate test files and `manifest.json`, kept apart from the development set |
| `test_cases/archive_v1/` | Original versions of two conformant files that were later rewritten |
| `results/` | Saved evaluation outputs (`.json`, `.md`) |
| `sample_codebase/`, `new_submission.py` | Small toy codebase and a deliberately bad submission from Milestone 1 |

## Setup

```
pip install -r requirements.txt
```

Get a free API key at https://console.groq.com and set it for the current
PowerShell window (it is never stored in the project):

```
$env:GROQ_API_KEY="your_key_here"
```

The model name is a setting, so it survives Groq retiring a model:

```
$env:GROQ_MODEL="openai/gpt-oss-120b"
```

The system prompt has two versions, chosen with `CODELENS_PROMPT` (`v1` is the default,
`v2` is the stricter rewrite described under Prompt iteration). Any other value raises an error.

```
$env:CODELENS_PROMPT="v2"
```

## Usage

Review one file against a codebase:

```
python run_llm.py path\to\file_to_review.py --repo path\to\repo --top-k 5
```

Other options:

```
python run_llm.py                  # review new_submission.py against sample_codebase/
python run_llm.py --mock           # offline plumbing test, no API key needed
python run_llm.py file.py --plain  # direct Groq call without LangChain, for comparison
```

`--mock` uses a fake model that cites one real and one made-up source. A correct
run shows 1 kept comment and 1 dropped comment.

## Evaluation

`evaluate.py` runs every file in `test_cases/` through CodeLens and through a
baseline (the same LLM with no retrieved codebase context), then scores both.

```
python evaluate.py --repo E:\Documents\Financial-Market-Risk-Analysis
python evaluate.py --rescore results\eval_20261118_141430.json
```

`--rescore` re-scores a saved run with the current scoring rules and makes no LLM calls.

**Corpus.** The codebase being indexed is `Financial-Market-Risk-Analysis`
(12 Python files, 51 functions). It was chosen because its conventions differ from
generic best practice: all names are snake_case, there are no docstrings, no type
hints, and functions return NaN or an empty Series on failure instead of raising.
A generic reviewer will flag all of that. A codebase-aware one should not.

**Test set.** 15 files. 5 are conformant (should get no comments) and 10 are
deviant, with 14 seeded violations in total: function naming, parameter naming,
vague parameter names, raising instead of returning NaN, bare `except`, the
`logging` module, `.format()` instead of f-strings, and `from numpy import *`.

**Metrics.**
- Recall: how many of the 14 seeded violations were caught.
- False alarms: how many conformant files got at least one comment.
- Citation validity: share of comments citing a function that was really retrieved.

**Results, first run** (18 Nov 2026, top-k 5, prompt v1, Groq `openai/gpt-oss-120b`):

| | CodeLens | Baseline (no retrieval) |
|---|---|---|
| Recall of seeded violations | 11/14 (79%) | 5/14 (36%) |
| Conformant files wrongly flagged | 2/5 | 5/5 |
| Comments on the 5 conformant files | 4 | 25 |
| Comments on deviant files | 18 (11 on target) | 51 (5 on target) |
| Citation validity | 22/22 (100%) | n/a |

This run used the original conformant files. CodeLens missed three violations:
`.format()` in `deviant_8`, and the vague parameter name and the raised exception in
`deviant_10`. It commented on two conformant files (`conformant_2`: 1 comment,
`conformant_5`: 3 comments).

**Results, three later runs** (same day, prompt v1, top-k 5, after the conformant files
were rewritten; scored with the same rules):

| Run | CodeLens recall | Conformant files flagged | Baseline recall | Baseline comments on conformant |
|---|---|---|---|---|
| `eval_20261118_154643` | 10/14 | 0/5 | 6/14 | 29 |
| `eval_20261118_155345` | 11/14 | 2/5 | 6/14 | 27 |
| `eval_20261118_160511` | 11/14 | 0/5 | 4/14 | 30 |

Across these runs CodeLens scored 10 to 11 out of 14, and the baseline 4 to 6 out of 14.
The baseline flagged all 5 conformant files every time. `deviant_10` (vague parameter
names, raised exception) was missed in all three runs and `deviant_9` (`import *`) in
the last two. Which violations are missed changes from run to run, so the same prompt
does not give the same answer twice.

**The three runs are not on an identical test set.** `conformant_4.py` was rewritten
between the second and third run (it returned a bool where the other `check_*`
functions return a status dict). The drop from 2/5 to 0/5 flagged files in the third
run is partly that fix and should not be read as the prompt improving.

**How scoring works.** "Caught" is decided by keyword rules in `evaluate.py`, so it is
approximate. Two of those rules (`raises_exception`, `logging_module`) were tightened
after the first run, because the original rules gave the baseline credit for comments
that mentioned the keyword without saying it conflicts with the codebase. Under the
original rules the baseline scored 8/14. The tightened rules agree with a by-hand
reading of every comment in `results/eval_20261118_141430.md`, which gives 5/14.
CodeLens scored 11/14 under both. `--rescore` of the first run reproduces the numbers
in the table above (11/14 and 5/14).

**What the results show.**
- The baseline spends most of its comments on generic advice (missing docstrings,
  type hints, input validation) that this repo does not follow, so it flags every
  conformant file. It also missed the camelCase names in `deviant_1` and `deviant_2`
  in the first run.
- CodeLens stays silent on most conformant files and does not suggest docstrings
  or type hints against this repo. The same pipeline does flag them when it indexes
  the toy `sample_codebase/`, which does use them. The review follows whichever
  codebase it is given.

**Problems found, reported as they are.**
- Conformant files in the first test set were not really conformant. `conformant_5`
  returned a scalar where every `forecast_*` function in the repo returns a Series of
  length `horizon`, so that comment was correct. Its second comment, about printing
  the error, was a real false alarm: only a minority of `except` blocks in the repo
  print. `conformant_2` returned a scalar where the neighbouring metrics return a
  Series. Both were rewritten to follow the repo's actual patterns, and later
  `conformant_4` was rewritten as well. The originals of the first two are in
  `test_cases/archive_v1/`. The first-run table above is from the original files.
- Because those files were changed after results were seen, the development set is
  no longer a clean test. Its numbers are best read as a development measurement.
- The model compares too literally to a single retrieved neighbour. It flags any
  difference from one reference, even when the new function computes something else.
- 100% citation validity only proves the cited function was retrieved. It does not
  prove the function supports the comment.
- A single run is not conclusive because the model varies between runs (see the range above).

**Held-out set.** `heldout_cases/` holds 8 further files (4 conformant, 4 deviant with
6 seeded violations: function naming, parameter naming, vague parameters, raising
instead of returning NaN, bare `except`, `.format()`). It was written separately from
the development set so that a prompt tuned on `test_cases/` can be checked on files it
was not tuned on. It is run with `python evaluate.py --repo <repo> --cases heldout_cases`.
No results from it are reported here.

## Prompt iteration

With the toy codebase and the first prompt, all 5 comments cited the same reference
(`cancel_order`), even where another function was the better example. Adding a rule
to cite the reference that best demonstrates each convention, and to check function
name, parameter names, type hints, docstring, exception type and error message
format explicitly, produced 6 comments with varied citations, including the error
message format comment that had disappeared.

A second version of the prompt (`v2`, selected with `CODELENS_PROMPT=v2`) was written
after reading where `v1` went wrong on the development set. It treats something as a
convention only if at least 3 of the retrieved reference functions follow it, tells the
model not to flag differences in what a function computes, and gives an explicit
checklist (vague single-letter names, bare `except`, raising versus returning a default,
message format, string formatting, imports, `logging` versus `print`). Because `v2` was
written with the development-set mistakes in view, its development-set score would be
optimistic, which is the reason the held-out set exists. The runs reported above all use
`v1`. `v2` needs at least 5 retrieved references to be meaningful, so use `--top-k 5`
(the default in `evaluate.py`; `run_llm.py` defaults to 3).

## Syllabus coverage in the current build

| Topic | Where |
|---|---|
| Large language models, prompt engineering (Unit III) | Review prompt in `llm_review.py`, prompt iteration above |
| LangChain (Unit III) | Retriever, prompt template, chat model, parser and validator in `chain.py` |
| Transformers (Unit III) | MiniLM sentence embeddings used for retrieval, ChatGroq for generation |
| Explainability (Unit IV) | Every comment cites the existing function it is based on, and citations are validated |

## Known limits of the current build

- Python only.
- Scoring is keyword-based, so a comment can be credited or missed by wording alone.
- The development test files were revised after results were seen (see above).
- MiniLM is trained on general text, not code, and reads roughly the first 256 tokens
  of each function.
- A long file under review is embedded as one block, so retrieval can miss parts of it.
- The LLM can be wrong. Its comments are suggestions for a human reviewer, not verdicts.