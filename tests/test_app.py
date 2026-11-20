"""Offline check of the Streamlit app: runs it with Streamlit's test runner in demo mode,
with a fake embedding function so nothing is downloaded and no API key is needed."""
import hashlib
import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fake_embed(texts):
    import numpy as np
    out = []
    for t in texts:
        v = np.zeros(384)
        for w in t.split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
        out.append((v / (np.linalg.norm(v) + 1e-9)).tolist())
    return out


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import chain
    import codelens
    monkeypatch.setattr(codelens, "embed_texts", fake_embed)
    monkeypatch.setattr(chain, "embed_texts", fake_embed)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
    at.session_state["folder"] = os.path.join(ROOT, "sample_codebase")
    return at.run()


def md(at):
    return " ".join(m.value for m in at.markdown)


def test_starts_without_errors(app):
    assert not app.exception
    assert "Review code the way a teammate" in md(app)


def test_review_shows_note_and_drops_invented_citation(app):
    app.text_area(key="code_input").set_value("def computeRollingVar(Returns, Window=21):\n    return Returns.rolling(Window).var()")
    next(b for b in app.button if b.key == "run").click().run()
    assert not app.exception
    text = md(app)
    assert 'class="note sel"' in text          # one kept note is selected
    assert "1 note dropped" in text            # the made-up citation was rejected
    assert "Note A is based on" in text        # evidence column shows the cited function


def test_empty_code_gives_message_not_crash(app):
    next(b for b in app.button if b.key == "run").click().run()
    assert not app.exception
    assert "Paste some code first" in md(app)


def test_missing_folder_is_reported(app):
    app.session_state["folder"] = os.path.join(ROOT, "no_such_folder")
    app.text_area(key="code_input").set_value("def f(x):\n    return x")
    next(b for b in app.button if b.key == "run").click().run()
    assert not app.exception
    assert "Folder not found" in md(app)
    

def test_new_review_clears_result(app):
    app.text_area(key="code_input").set_value("def computeRollingVar(Returns, Window=21):\n    return Returns.rolling(Window).var()")
    next(b for b in app.button if b.key == "run").click().run()
    assert app.session_state["result"] is not None
    next(b for b in app.button if b.key == "new_btn").click().run()
    assert not app.exception
    assert app.session_state["result"] is None
    assert app.session_state["code_input"] == ""
    assert "Review code the way a teammate" in md(app)