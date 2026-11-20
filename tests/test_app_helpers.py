"""Unit tests for helper functions in app.py. app.py runs the Streamlit page when imported, so the
functions under test are pulled out of its source with the ast module and run on their own."""
import ast
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load(names, extra=None):
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))]
    ns = {"re": re, "os": os, "shutil": shutil, "subprocess": subprocess,
          "urllib": urllib, "zipfile": zipfile, **(extra or {})}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "app.py", "exec"), ns)
    return ns


# --------------------------------------------------------------- evidence_lines
@pytest.fixture(scope="module")
def ev():
    from evaluate import CATEGORY_RULES
    return load({"CATEGORY_ORDER", "categorize", "evidence_lines"}, {"CATEGORY_RULES": CATEGORY_RULES})["evidence_lines"]


REF = '''def load_prices(path):
    """Load prices from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file at {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Read failed: {e}")
        return None'''


def test_error_note_highlights_raise_line(ev):
    note = {"issue": "Raises exception", "comment": "The reference raises an exception instead of returning None."}
    assert ev(note, REF) == [4]


def test_naming_note_highlights_def_line(ev):
    note = {"issue": "Function name style", "comment": "Use snake_case for function names, not camelCase."}
    assert ev(note, REF) == [1]


def test_return_note_highlights_return_lines(ev):
    note = {"issue": "Missing return statement", "comment": "Reference functions always return a value."}
    assert ev(note, REF) == [6, 9]


def test_no_match_falls_back_to_def_line(ev):
    note = {"issue": "Uses logging module", "comment": "Use logging instead of print."}
    ref = "def f(x):\n    return x"
    assert ev(note, ref) == [1]


# ----------------------------------------------------------------- fetch_github
@pytest.fixture()
def gh(tmp_path):
    ns = load({"GITHUB_RE", "fetch_github"}, {"CACHE_DIR": str(tmp_path)})
    return ns, tmp_path


def test_bad_link_is_rejected(gh):
    ns, _ = gh
    path, err = ns["fetch_github"]("https://gitlab.com/a/b")
    assert path is None and "github.com" in err


def test_finished_copy_is_reused_without_download(gh, monkeypatch):
    ns, tmp = gh
    dest = tmp / "owner__repo"
    dest.mkdir()
    (dest / ".codelens_ok").write_text("")

    def no_network(*a, **k):
        raise AssertionError("should not clone a finished copy")
    monkeypatch.setattr(subprocess, "run", no_network)
    assert ns["fetch_github"]("https://github.com/owner/repo") == (str(dest), None)


def test_half_finished_copy_is_downloaded_again(gh, monkeypatch):
    ns, tmp = gh
    dest = tmp / "owner__repo"
    dest.mkdir()
    (dest / "partial.py").write_text("")             # left over from a broken clone, no marker
    calls = []

    def fake_clone(cmd, **k):
        calls.append(cmd)
        os.makedirs(cmd[-1], exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_clone)
    path, err = ns["fetch_github"]("https://github.com/owner/repo")
    assert err is None and len(calls) == 1
    assert not (dest / "partial.py").exists()        # the broken copy was removed first
    assert (dest / ".codelens_ok").exists()          # and the new copy is marked finished
