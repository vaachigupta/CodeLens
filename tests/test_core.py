"""Offline tests: no API key, no embedding model, no network.  Run:  python -m pytest -q"""
import json, os, sys, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from codelens import collect_repo_chunks, extract_function_chunks
from llm_review import _norm, parse_and_validate


def write(tmp_path, name, code):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(code), encoding="utf-8")
    return str(p)


# ----------------------------------------------------------- chunker
def test_chunker_finds_functions_and_methods(tmp_path):
    f = write(tmp_path, "a.py", """
        def top(x): return x
        class K:
            def m(self): return 1
            async def am(self): return 2
    """)
    ids = [c["id"].split("::")[1] for c in extract_function_chunks(f)]
    assert ids == ["top", "K.m", "K.am"]


def test_chunker_metadata(tmp_path):
    f = write(tmp_path, "a.py", '''
        def good_name(x):
            """doc"""
            return x
        def badName(x): return x
    ''')
    c = {c["function_name"]: c for c in extract_function_chunks(f)}
    assert c["good_name"]["has_docstring"] and c["good_name"]["is_snake_case"]
    assert not c["badName"]["has_docstring"] and not c["badName"]["is_snake_case"]
    assert c["badName"]["line"] == 5 and c["badName"]["source"].startswith("def badName")


def test_collect_skips_venv_hidden_broken_and_excluded(tmp_path):
    write(tmp_path, "keep.py", "def keep(): pass\n")
    write(tmp_path, "venv/lib.py", "def lib(): pass\n")
    write(tmp_path, ".git/x.py", "def hidden(): pass\n")
    write(tmp_path, "broken.py", "def (:\n")
    target = write(tmp_path, "target.py", "def target(): pass\n")
    chunks, n, skipped = collect_repo_chunks(str(tmp_path), exclude=target)
    assert [c["function_name"] for c in chunks] == ["keep"]
    assert n == 1 and len(skipped) == 1 and skipped[0].endswith("broken.py")


def test_duplicate_function_names_get_unique_ids(tmp_path):
    write(tmp_path, "a.py", "def f(): pass\ndef f(): pass\n")
    chunks, _, _ = collect_repo_chunks(str(tmp_path))
    assert len({c["id"] for c in chunks}) == 2


# ----------------------------------------------- citation validation
def test_norm_treats_slashes_alike():
    assert _norm("src\\metrics.py::f") == _norm("src/metrics.py::f")


def _raw(*cites):
    return json.dumps({"comments": [{"issue": "i", "comment": "c", "cited_source": c} for c in cites]})


def test_validator_keeps_real_and_drops_invented():
    kept, dropped = parse_and_validate(_raw("src/a.py::f", "nowhere.py::x"), ["src/a.py::f"])
    assert len(kept) == 1 and len(dropped) == 1


def test_validator_accepts_windows_ids():
    kept, _ = parse_and_validate(_raw("src/a.py::f"), ["src\\a.py::f"])
    assert len(kept) == 1


def test_validator_drops_empty_comment_and_handles_bad_json():
    raw = json.dumps({"comments": [{"issue": "i", "comment": "", "cited_source": "a::f"}]})
    assert parse_and_validate(raw, ["a::f"]) == ([], json.loads(raw)["comments"])
    kept, dropped = parse_and_validate("not json", ["a::f"])
    assert kept == [] and "error" in dropped[0]


def test_empty_comment_list_is_a_valid_silent_review():
    assert parse_and_validate(json.dumps({"comments": []}), ["a::f"]) == ([], [])


# -------------------------------------------------------- chain + redaction
class StaticRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.docs


def test_chain_keeps_real_citation_and_redacts_secrets_before_llm():
    from chain import build_chain

    key = "gsk_" + "Ab1Cd2Ef" * 4
    ref = Document(page_content=f'def get_x():\n    api_key = "{key}"\n', metadata={"id": "src/a.py::get_x"})
    seen = {}

    def fake_llm(prompt_value):
        seen["prompt"] = prompt_value.to_string()
        return json.dumps({"comments": [
            {"issue": "n", "comment": "c", "cited_source": "src/a.py::get_x"},
            {"issue": "m", "comment": "c", "cited_source": "fake.py::y"}]})

    chain = build_chain(StaticRetriever(docs=[ref]), RunnableLambda(fake_llm))
    out = chain.invoke({"new_code": f'def getX():\n    token = "{key}"\n'})
    assert key not in seen["prompt"]
    assert "<REDACTED" in seen["prompt"]
    assert len(out["kept"]) == 1 and len(out["dropped"]) == 1
    assert len(out["redactions"]) >= 2  # one in the new code, one in the reference


# ------------------------------------------------------------ scoring rules
def test_scoring_rules_on_clear_examples():
    from evaluate import CATEGORY_RULES as R
    assert R["function_naming"]("the function name uses camelcase")
    assert R["param_naming"]("parameter names use pascalcase")
    assert R["bare_except"]("uses a bare except")
    assert R["import_star"]("from numpy import * is a wildcard import")
    assert not R["raises_exception"]("will raise an attributeerror when x is none")
    assert not R["function_naming"]("add a docstring")
