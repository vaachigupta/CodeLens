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
[Redactor]    secrets (API keys, tokens, passwords) in the code and references replaced by <REDACTED:type>
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
5. **Secret redaction** (`redact.py`, called from `chain.py`): before anything is sent to
   the LLM, the new code and the retrieved references are scanned for API keys, tokens,
   private keys, passwords in URLs and secret-looking assignments, and each is replaced
   by `<REDACTED:type>`. The chain's output lists what was redacted, with line numbers
   and a masked preview (never the secret itself).
6. **Review** (`chain.py`, `llm_review.py`): a `ChatPromptTemplate` feeds the references and
   the new code to ChatGroq, which replies in JSON.
7. **Citation check** (`llm_review.py`): comments citing anything outside the retrieved
   set are dropped, which stops the model from inventing sources.

## Files

| File | Purpose |
|---|---|
| `codelens.py` | Chunking, embeddings, Chroma vector store, retrieval |
| `llm_review.py` | System prompt, Groq call, JSON parsing, citation validation |
| `chain.py` | LangChain chain: retriever, prompt, model, parser, validator (plus an offline mock model) |
| `app.py`, `.streamlit/config.toml` | Streamlit web interface (three columns: your code, review notes, the cited function) |
| `run_llm.py` | Command line runner for reviewing one file |
| `evaluate.py` | Evaluation harness: CodeLens versus a no-retrieval baseline |
| `redact.py` | Secret redaction applied before code reaches the LLM |
| `retrieval_eval.py` | Evaluates the retrieval stage alone (no LLM, no API key) |
| `autoencoder_novelty.py` | Small numpy autoencoder that scores how unlike the codebase a function is |
| `tests/` | Offline tests (`python -m pytest -q`): chunker, citation check, redaction, chain, scoring rules, autoencoder, web interface |
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

Web interface:

```
streamlit run app.py
```

Paste code (or pick a sample), choose a codebase (a local folder or a public GitHub link) from the
top bar, and press Review code. The page has three columns. The left one shows your code with the lines
a note is about highlighted. The middle one lists the review notes. Clicking "Show this one" on a note
selects it, and the right column then shows the existing function that note is based on, with line
numbers. Notes that cite a function that was not retrieved are rejected and counted. Secrets found in the
code are listed as redacted. Settings holds top-k, the prompt version (v1 or v2), a demo mode that uses a
fake model so the interface can be tried without an API key (its comments are not real reviews), and an
option to show the same model without the codebase next to the review. There is a Light and a Dark
theme. Which lines a note highlights is decided by simple pattern rules, not by the model: on your code
(for example the `except:` line for a bare-except note) and on the cited function (for example its
`raise` line for an error-handling note). After a review, Edit code keeps your code for another try,
New review clears the page, and Export review saves the notes as Markdown. The comparison without the
codebase is only shown if it was switched on before the review was run. A GitHub repository is fetched
with a shallow `git clone`; if git is missing, fails or takes over 60 seconds, the repository is
downloaded as a zip instead. A copy is only reused once it has finished downloading, so a broken clone
is fetched again. Each codebase is indexed once per session, so later reviews against it are faster.

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

Other tools (none of them need an API key):

```
python -m pytest -q                      # 33 offline tests
python redact.py path\to\file.py         # show the code with secrets redacted, and what was found
python retrieval_eval.py --repo <repo>   # retrieval stage on its own
python autoencoder_novelty.py --repo <repo> --foreign <other_repo>
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

**Results on the revised development set** (same day, top-k 5, all scored with the same
rules; `conformant_2`, `conformant_4` and `conformant_5` had been rewritten by then):

| Run | Prompt | CodeLens recall | Conformant files flagged | Baseline recall |
|---|---|---|---|---|
| `eval_20261118_154643` | v1 | 10/14 | 0/5 | 6/14 |
| `eval_20261118_155345` | v1 | 11/14 | 2/5 | 6/14 |
| `eval_20261118_160511` | v1 | 11/14 | 0/5 | 4/14 |
| `eval_20261119_114808` | v1 | 9/14 | 0/5 | not run |
| `eval_20261119_115231` | v1 | 12/14 | 0/5 | not run |
| `eval_20261119_114338` | v2 | 12/14 | 0/5 | not run |

With prompt v1 the recall ranged from 9 to 12 out of 14 over five runs, so the
same prompt does not give the same answer twice. The baseline (same LLM, no
retrieval) scored 4 to 6 out of 14 and flagged all 5 conformant files in every run it was
run. Prompt v2 scored 12/14 in its single development run, which is inside the range v1
produced, so on the development set v2 is not shown to be better on recall. `deviant_7`
(`logging`) was caught in 2 of the 6 runs and `deviant_8` (`.format()`) in 3 of 6.

**The first two runs in this table used an older `conformant_4.py`.** It was rewritten
between `eval_20261118_155345` and `eval_20261118_160511` (it returned a bool where the other
`check_*` functions return a status dict). The drop from 2/5 to 0/5 flagged files from the
third run on is partly that fix.

**Results on the held-out set** (8 files: 4 conformant, 4 deviant, 6 seeded violations;
top-k 5; prompt v1 run twice, v2 three times, the baseline once; the table below shows the first run of each prompt):

| | CodeLens, prompt v1 | CodeLens, prompt v2 | Baseline (no retrieval) |
|---|---|---|---|
| Recall of seeded violations | 6/6 | 6/6 | 3/6 |
| Conformant files wrongly flagged | 2/4 (4 comments) | 0/4 (0 comments) | 4/4 (24 comments) |
| Comments on deviant files | 8 (6 on target) | 6 (6 on target) | 24 (3 on target) |
| Citation validity | 12/12 | 6/6 | n/a |

A second held-out run with prompt v1 (`eval_20261119_133321`, no baseline) scored 5/6 and flagged 2/4
conformant files (`conformant_h2`: 2 comments, `conformant_h4`: 2 comments). It missed the `.format()` in
`deviant_h4`: its second comment is about the printed message lacking the exception, not the string
format. So prompt v1 scored 5/6 and 6/6 on the same held-out files and flagged the same two conformant
files both times, which shows the run-to-run spread on this set too.

In the table above, the first v1 run and the first v2 run caught every seeded violation in the held-out set. Prompt v2 produced no
comments on conformant files and no off-target comments, where v1 produced 4 comments on
two conformant files (`conformant_h2`, `conformant_h4`). This is one run each on 8 files,
so it is an indication and not a proven difference. Some of v1's held-out comments may also
be legitimate: for example `conformant_h4` omits the `annualize_factor` parameter that the
repo's `forecast_*` functions take. The held-out files were not edited after the results
were seen, so this is reported as it came out.

**Prompt v2, all three held-out runs:**

| Run | Recall | Conformant files flagged | Comments on deviant files | Citation validity |
|---|---|---|---|---|
| 1 | 6/6 | 0/4 | 6 (6 on target) | 6/6 |
| `eval_20261120_104306` | 6/6 | 0/4 | 6 (6 on target) | 6/6 |
| `eval_20261120_104617` | 5/6 | 0/4 | 5 (5 on target) | 5/7 (2 dropped: malformed citation) |

Over three held-out runs, prompt v2 caught 6/6, 6/6 and 5/6 seeded violations and made no comment on any conformant file. Prompt v1 flagged 2 of the 4 conformant files in both of its runs. The single miss (`eval_20261120_104617`, `deviant_h2`) was a citation formatting error: the model cited `src/detect_market_regime`, leaving out the file name (the retrieved ID is `src/regime_detection.py::detect_market_regime`), so the validator dropped both of its comments. The first of those was the correct finding (raising instead of returning a default); the second made a wrong claim about the repo's string style. The check requires an exact match, so a correct comment with a slightly wrong citation is lost. Three runs on 8 files is still a small sample.

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

**Retrieval on its own.** `retrieval_eval.py` checks the retrieval stage without any LLM.
A retrieved function counts as relevant if it has the same name family as the new function
(the first word: `calculate`, `get`, `forecast`, `check`, `load`). It compares MiniLM
retrieval in Chroma with plain TF-IDF and with chance.

| Test set | Method | Hit@1 | Hit@3 | Hit@5 | Precision@5 |
|---|---|---|---|---|---|
| Development (15 files) | MiniLM | 10/15 | 13/15 | 14/15 | 45% |
| Development (15 files) | TF-IDF | 7/15 | 12/15 | 12/15 | 35% |
| Held-out (8 files) | MiniLM | 4/8 | 6/8 | 7/8 | 45% |
| Held-out (8 files) | TF-IDF | 5/8 | 6/8 | 6/8 | 32% |

Chance precision is 21% (development) and 18% (held-out). MiniLM is well above chance and
at least as good as TF-IDF from k=3; on the held-out set TF-IDF was ahead at k=1 (5/8
against 4/8). With 15 and 8 files, one file moves a number by 7 to 12 points. Top-k 5 was
chosen because Hit@5 is higher than Hit@3 on both sets. The ground truth is a naming proxy,
not a judgement of which function is the best reference. MiniLM matches on what the code does,
not on how it is named: the camelCase files `computeRollingVar` and `GetVolatility` did not
retrieve a same-family function first. Retrieval quality did not predict which violations the
LLM caught: `deviant_10` had all four of its violations caught in the prompt v2 run even though none
of its retrieved functions share its family.

**Autoencoder novelty score.** `autoencoder_novelty.py` trains a small autoencoder
(384 -> 16 -> 384, written in numpy, trained with Adam) on the MiniLM embeddings of the
codebase's functions. A function it reconstructs badly is unlike the codebase. Scores are
calibrated with 5-fold cross-validation (mean error on training functions 0.00043, on
held-out folds 0.00195). On `Financial-Market-Risk-Analysis` it separates off-topic code well:
17 functions from MediSage scored a mean novelty of 0.95 against 0.49 for the repo's own
functions (AUC 0.96). It does not separate convention violations from conformant files
(AUC 0.64 on 15 files, within noise), because embeddings capture what code does, not how it is
styled. It is a standalone script and is not part of the review chain.

**Secret redaction.** `redact.py` is covered by 7 tests, including a chain test that
checks a secret placed in both the new code and a reference never appears in the prompt
sent to the model.

**Tests.** 33 offline tests pass (`python -m pytest -q`): the chunker (functions, methods,
skipped folders and broken files, duplicate names), citation validation (including
Windows `\` against `/` paths, bad JSON, empty comment lists), redaction, the full chain with
a fake model, spot checks of the scoring rules, the autoencoder, and the web interface (run with
Streamlit's test runner in demo mode: a review shows a note, the made-up citation is dropped, empty code
and a missing folder give a message and not a crash, New review clears the page), and helper functions
of the web interface (which line of a cited function is highlighted for a note, and that a finished
GitHub download is reused while a half-finished one is fetched again).

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
optimistic, which is the reason the held-out set exists. The development-set runs
and the held-out runs above report both versions. `v2` is meant to be used with `--top-k 5` (the default in `evaluate.py`; `run_llm.py`
defaults to 3).

## Syllabus coverage in the current build

| Topic | Where |
|---|---|
| Large language models, prompt engineering (Unit III) | Review prompt in `llm_review.py`, prompt iteration above |
| LangChain (Unit III) | Retriever, prompt template, chat model, parser and validator in `chain.py` |
| Transformers (Unit III) | MiniLM sentence embeddings used for retrieval, ChatGroq for generation |
| Explainability (Unit IV) | Every comment cites the existing function it is based on, and citations are validated |
| Responsible AI (Unit IV) | `redact.py` removes secrets from code before it is sent to an external model |
| Autoencoders (Unit I) | `autoencoder_novelty.py`: numpy autoencoder on code embeddings, evaluated with cross-validation |
| Prompt engineering (Unit III) | Prompt v1 against v2, compared on development and held-out sets |

## Known limits of the current build

- Python only.
- In the web interface the highlighted lines, in your code and in the cited function, are chosen by
  pattern rules, so a note can highlight the `def` line when no better line is found.
- Scoring is keyword-based, so a comment can be credited or missed by wording alone.
- The development test files were revised after results were seen (see above), so they
  are no longer a clean test. The held-out set was not edited.
- The test sets are small (15 and 8 files). Prompt v2 was run three times on the held-out set and
  once on the development set, so differences of one or two files are not conclusive.
- Only the seeded violations are scored. A comment that is correct but about something else
  (for example `deviant_7` returning a bool where the repo's `check_*` functions return a
  dict) counts as off-target. Whether a "conformant" file is truly conformant is itself a
  judgement; two of the original ones were not.
- Citation validity checks that the cited function was retrieved, not that the claim about
  it is true. A few comments contain small misquotes (for example `Return` for the parameter `Returns`).
- MiniLM is trained on general text, not code, and reads roughly the first 256 tokens
  of each function.
- A long file under review is embedded as one block, so retrieval can miss parts of it.
- The LLM can be wrong. Its comments are suggestions for a human reviewer, not verdicts.
- The citation check requires an exact match with a retrieved ID. A comment that names the right function with a malformed path (for example without the file name) is dropped, which cost one catch in the third v2 held-out run. Accepting a unique name match was not added, because it would be a change made after seeing held-out results.