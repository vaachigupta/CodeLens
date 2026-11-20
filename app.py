"""CodeLens web UI.   Run:  streamlit run app.py

Three columns: your code, review notes, and the function from your codebase that each note
is based on. Click a note to see which lines it is about and the function it cites.

Secrets are redacted before the LLM sees anything, and comments citing something that was
not retrieved are rejected (see chain.py).
"""
import glob
import hashlib
import html
import keyword
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile

import chromadb
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

import chain as chain_mod
from chain import CodebaseRetriever, build_chain, get_llm, get_mock_llm
from codelens import collect_repo_chunks, embed_texts
from evaluate import CATEGORY_RULES, build_baseline_chain, get_mock_baseline_llm
from llm_review import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2, _norm

st.set_page_config(page_title="CodeLens", layout="wide", initial_sidebar_state="collapsed")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "repos_cache")      # cloned GitHub repos live here
MAX_FUNCS = 2000                                    # keep indexing time reasonable
DEFAULT_REPO = r"E:\Documents\Financial-Market-Risk-Analysis"
if not os.path.isdir(DEFAULT_REPO):
    DEFAULT_REPO = os.path.join(HERE, "sample_codebase")

# ------------------------------------------------------------------------------ theme
# Rose and gold. Every colour is a CSS variable so the Light / Dark switch only swaps this table.
THEMES = {
    "light": dict(page="#fff6f9", panel="#fff6f9", deep="#4a1530", ev="#f0d3df", notes="#fae9f0",
                  ink="#2a1220", muted="#7a5668", rule="#ecd0db", field="#fffafc", btn="#fffafc", accent="#cf2e67",
                  accent_fg="#ffffff", accent2="#cf2e67", hot="#a87900", gold="#ffd04d", head="#2a1220",
                  hl="rgba(255,208,77,.55)", hl_sel="rgba(255,196,40,.85)", kw="#a8236b", str="#1b7a55",
                  num="#c2500f", cmt="#a98a99", tb_bg="#fff6f9", tb_fg="#2a1220", tb_mute="#7a5668",
                  tb_chip="#fffafc", tb_line="#ecd0db", evi_panel="#fff3f7", evi_ink="#2a1220",
                  evi_muted="#7a5668", evi_rule="#ecd5de", evi_head="#2a1220"),
    "dark": dict(page="#4f1230", panel="#661a40", deep="#8e1f55", ev="#430e29", notes="#4f1230",
                 ink="#fff0f5", muted="#f0b6cc", rule="#8a2e58", field="#3f0c26", btn="#661a40", accent="#ff7aa8",
                 accent_fg="#3a0a22", accent2="#ffd04d", hot="#ffd04d", gold="#ffd04d", head="#ffd9e6",
                 hl="rgba(255,208,77,.24)", hl_sel="rgba(255,208,77,.46)", kw="#ffb3d0", str="#9be8b8",
                 num="#ffc27a", cmt="#d08aa6", tb_bg="#8e1f55", tb_fg="#ffffff", tb_mute="rgba(255,255,255,.72)",
                 tb_chip="rgba(255,255,255,.1)", tb_line="rgba(255,255,255,.28)", evi_panel="#34091f",
                 evi_ink="#eef4f3", evi_muted="rgba(238,244,243,.68)", evi_rule="rgba(255,255,255,.16)",
                 evi_head="#ffffff"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=JetBrains+Mono:wght@400;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,500&display=swap');
:root{__VARS__
  --headf:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
  --mono:"JetBrains Mono","Cascadia Mono",ui-monospace,Consolas,monospace;
  --radius:6px;}
/* ---- Streamlit chrome off, full-bleed page ---- */
header[data-testid="stHeader"], [data-testid="stToolbar"], .stAppDeployButton, footer {display:none !important;}
.stApp {background:var(--page); color:var(--ink); font-family:var(--serif); font-size:16px;}
.block-container {padding:0 !important; max-width:100% !important;}
[data-testid="stVerticalBlock"] {gap:0;}
.stApp p, .stApp label, .stApp span, .stApp li {color:inherit;}
h1,h2,h3 {font-family:var(--headf); color:var(--head);}
/* ---- top bar ---- */
.st-key-topbar {background:var(--tb_bg); color:var(--tb_fg); padding:0 22px; min-height:60px;
  border-bottom:1px solid var(--rule); box-shadow:inset 0 -3px 0 var(--accent);}
.st-key-topbar [data-testid="stHorizontalBlock"] {align-items:center; gap:10px; min-height:60px;}
.brand {display:flex; align-items:center; gap:9px; font-family:var(--headf); font-size:24px; font-weight:700;
  letter-spacing:-.01em; color:var(--tb_fg);}
.brand .tag {font-family:var(--serif); font-style:italic; font-weight:400; font-size:14px; color:var(--tb_mute);
  margin-left:10px; letter-spacing:0;}
.demo-note {font-size:14px; color:var(--tb_mute); font-style:italic; text-align:right;}
.st-key-topbar button, .st-key-topbar [data-testid="stPopover"] button {background:var(--tb_chip) !important;
  border:1px solid var(--tb_line) !important; color:var(--tb_fg) !important; border-radius:var(--radius) !important;
  height:38px; min-height:38px; padding:0 14px; font-family:var(--serif); font-size:15px;}
.st-key-topbar button p {color:var(--tb_fg) !important; font-size:15px;}
.st-key-topbar button:hover {border-color:var(--tb_mute) !important;}
/* ---- three columns ---- */
.st-key-cols [data-testid="stHorizontalBlock"] {gap:0 !important; align-items:stretch;}
.st-key-cols [data-testid="stColumn"] {padding:0; min-width:0;}
.st-key-col_code, .st-key-col_notes, .st-key-col_evi {min-height:calc(100vh - 60px); border-right:1px solid var(--rule);}
.st-key-col_code {background:var(--panel);}
.st-key-col_notes {background:var(--notes);}
.st-key-col_evi {background:var(--ev); border-right:0; --ink:var(--evi_ink); --muted:var(--evi_muted); --rule:var(--evi_rule);
  --head:var(--evi_head); --panel:var(--evi_panel); color:var(--evi_ink);}
.chead {display:flex; align-items:center; gap:11px; min-height:62px; padding:0 22px; border-bottom:1px solid var(--rule);}
.chead h2 {margin:0; padding:0 !important; line-height:1.2; font-size:24px; font-weight:700; letter-spacing:-.015em; white-space:nowrap;
  display:flex; align-items:center; gap:11px; color:var(--head);}
.chead h2:before {content:""; width:5px; height:1.05em; border-radius:2px; background:var(--accent);}

.st-key-run_body, .st-key-notes_body, .st-key-evi_body {padding:22px;}
.pad {padding:0 22px;}
[class*="st-key-head_"] {border-bottom:1px solid var(--rule); height:62px !important; min-height:62px !important; box-sizing:border-box; flex:none !important; padding:0 22px; display:flex !important; flex-direction:column; justify-content:center;}
[class*="st-key-head_"] [data-testid="stHorizontalBlock"] {width:100%; gap:8px !important; align-items:center;}
[class*="st-key-head_"] [data-testid="stColumn"]:last-child {display:flex; justify-content:flex-end;}
[class*="st-key-head_"] .chead {border:0; padding:0; min-height:0;}
.st-key-head_code {padding-right:40px !important;}
.st-key-head_code [data-testid="stColumn"]:last-child [data-testid="stElementContainer"] {margin-left:auto;}
.stApp code {background:transparent !important; color:var(--ink) !important; padding:0 !important;}
/* ---- buttons ---- */
.stButton button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] > div > button {
  background:var(--btn); border:1px solid var(--rule); color:var(--ink); border-radius:var(--radius);
  font-family:var(--serif); font-size:15px; min-height:34px; height:34px; padding:0 14px;}
.stButton button p {color:inherit; font-size:15px;}
.stButton button:hover {border-color:var(--muted);}
.stButton button[kind="primary"] {background:var(--accent); color:var(--accent_fg); border-color:var(--accent);
  font-weight:650;}
.stButton button[kind="primary"] p {color:var(--accent_fg);}
.stButton button:disabled {opacity:.45;}
.st-key-samples .stButton button {border-radius:999px; background:color-mix(in srgb,var(--accent) 9%,transparent);
  border-color:color-mix(in srgb,var(--accent) 35%,transparent); height:32px; min-height:32px;}
.st-key-samples {padding:11px 22px; border-bottom:1px solid var(--rule);}
.st-key-samples [data-testid="stHorizontalBlock"] {gap:8px !important; align-items:center;}
.st-key-samples [data-testid="stColumn"] {width:auto !important; flex:0 0 auto !important; min-width:0;}
.st-key-samples [data-testid="stPopover"] button {border-radius:999px;}
.lead {color:var(--muted); font-size:15px; white-space:nowrap;}
/* ---- inputs ---- */
textarea, input[type="text"], [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
  background:var(--field) !important; color:var(--ink) !important; border-radius:var(--radius);}
[data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stTextInputRootElement"] {background:var(--field) !important;}
textarea::placeholder, input::placeholder {color:var(--muted) !important; opacity:.85 !important;}
[data-testid="stPopoverBody"] .stButton button {background:var(--btn) !important; color:var(--ink) !important; border:1px solid var(--rule) !important;}
[data-testid="stPopoverBody"] .stButton button p {color:var(--ink) !important;}
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] {color:var(--muted);}
.stApp .st-key-code_box textarea {color:var(--ink) !important; font-family:var(--mono) !important; font-size:13.5px !important; line-height:1.75 !important;
  border:0 !important; border-radius:0 !important; background:transparent !important; padding:14px 22px !important;}
.stApp .st-key-code_box [data-baseweb="textarea"], .stApp .st-key-code_box [data-baseweb="base-input"],
.stApp .st-key-code_box [data-testid="stTextArea"] > div > div {background:transparent !important; border:0 !important; box-shadow:none !important;}
.stApp .st-key-code_box textarea, .stApp .st-key-code_box [data-testid="stTextAreaRootElement"] {background:transparent !important;}
[data-testid="stPopoverBody"], [data-baseweb="popover"] > div {background:var(--panel) !important; color:var(--ink);
  border:1px solid var(--rule); border-radius:8px; box-shadow:0 8px 28px rgba(20,33,41,.28);}
[data-testid="stPopoverBody"] p, [data-testid="stPopoverBody"] label, [data-testid="stPopoverBody"] span {color:var(--ink);}
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {color:var(--ink) !important;}
.stApp [data-testid="stExpander"] {border:0; background:transparent;}
.stApp [data-testid="stFileUploader"] section {background:var(--field); border:1px dashed var(--rule);}
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {gap:.7rem;}
[data-testid="stPopoverBody"] code {background:var(--field) !important; color:var(--ink) !important; padding:1px 5px !important;
  border:1px solid var(--rule); border-radius:4px;}
/* ---- code view ---- */
.code {font-family:var(--mono); font-size:13.5px; line-height:1.75; padding:14px 0 40px; overflow-x:auto;}
.ln {display:flex; align-items:baseline; padding-right:22px; position:relative;}
.ln .no {flex:none; width:52px; padding-right:14px; text-align:right; color:var(--muted); opacity:.7; user-select:none;}
.ln .src {white-space:pre; color:var(--ink);}
.ln .fn {flex:none; display:flex; gap:4px; margin-left:12px;}
.ln.marked {background-image:linear-gradient(var(--hl),var(--hl)); background-repeat:no-repeat; background-size:100% 100%;
  animation:sweep .52s ease-out both;}
.ln.marked.sel {background-image:linear-gradient(var(--hl_sel),var(--hl_sel));}
@keyframes sweep {0%{background-size:0% 100%} to{background-size:100% 100%}}
.foot {width:17px; height:17px; display:inline-grid; place-items:center; border-radius:50%; background:var(--accent);
  color:var(--accent_fg); font-family:var(--serif); font-style:italic; font-weight:600; font-size:12px; line-height:1;}
.foot.off {opacity:.7;}
.t-keyword{color:var(--kw)} .t-string{color:var(--str)} .t-number{color:var(--num)}
.t-comment{color:var(--cmt); font-style:italic} .t-defname{font-weight:700}
.banner {padding:10px 22px; font-size:15px; background:var(--hl); border-bottom:1px solid var(--rule); color:var(--ink);}
/* ---- notes ---- */
.empty {max-width:46ch;}
.empty p {line-height:1.6; color:var(--muted); font-size:17px; margin:0 0 14px;}
.empty .lede {font-family:var(--headf); color:var(--head); font-size:26px; line-height:1.15; font-weight:750;
  margin-bottom:18px; letter-spacing:-.02em;}
.summary {margin:0 0 18px; color:var(--muted); font-size:16px; line-height:1.5;}
.note {display:flex; gap:14px; padding:18px 12px 18px 14px; border-top:1px solid var(--rule);
  border-left:3px solid transparent;}
.note.sel {border-left:4px solid var(--accent); padding-left:13px; background:color-mix(in srgb,var(--accent) 9%,var(--panel));}
.note:last-of-type {border-bottom:1px solid var(--rule);}
.letter {flex:none; width:24px; height:24px; margin-top:2px; display:grid; place-items:center; border-radius:50%;
  background:var(--accent); color:var(--accent_fg); font-style:italic; font-weight:600; font-size:15px;}
.nbody {display:flex; flex-direction:column; gap:6px; min-width:0;}
.ntitle {color:var(--head); font-family:var(--headf); font-weight:700; font-size:20px; letter-spacing:-.01em;
  display:flex; gap:12px; align-items:baseline; flex-wrap:wrap;}
.cat {font-family:var(--serif); font-weight:500; font-size:14.5px; color:var(--accent); font-style:italic;}
.ntext {font-size:17px; line-height:1.55; color:var(--ink);}
.nsrc {font-size:14.5px; color:var(--muted);}
.nsrc code, code {font-family:var(--mono); font-size:.86em; color:var(--ink); background:transparent; padding:0;}
[class*="st-key-pick_"] .stButton button {height:28px; min-height:28px; font-size:14px; padding:0 10px; margin:-6px 0 0 0;
  background:transparent; color:var(--accent); border-color:transparent; text-decoration:underline; text-underline-offset:3px;}
.allclear {padding:18px 16px; border-left:3px solid var(--str); background:var(--panel);}
.allclear strong {font-size:18px;} .allclear p {margin:6px 0 0; color:var(--muted); font-size:16.5px;}
.generic {margin-top:28px; padding-top:18px; border-top:1px dashed var(--rule);}
.generic h3 {font-size:18px; margin:0;}
.generic ul {margin:10px 0; padding-left:18px; color:var(--muted); font-size:15.5px; line-height:1.5;}
.generic li {margin-bottom:6px;} .generic strong {color:var(--ink); font-weight:600;}
.muted {color:var(--muted); font-size:14.5px; margin:4px 0;}
.dropped {margin-top:22px; color:var(--muted); font-size:15px;}
.dropped summary, .consulted summary {cursor:pointer;}
.dropped p {margin:8px 0 0; line-height:1.5;}
.progress {list-style:none; margin:6px 0 0; padding:0;}
.progress li {display:flex; align-items:center; gap:12px; padding:9px 0; font-size:17px; color:var(--muted);}
.progress .dot {width:20px; height:20px; border-radius:50%; border:2px solid var(--rule); display:grid; place-items:center;}
.progress li.done {color:var(--ink);} .progress li.done .dot {background:var(--accent); border-color:var(--accent); color:var(--accent_fg);}
.progress li.now {color:var(--ink); font-weight:600;}
.progress li.now .dot {border-color:var(--accent); border-top-color:transparent; animation:spin .8s linear infinite;}
@keyframes spin {to{transform:rotate(360deg)}}
/* ---- evidence ---- */
.evi-for {margin:0; color:var(--muted); font-size:15px;}
.nsrc code {overflow-wrap:break-word;}
.evi-name {margin:2px 0 4px; color:var(--hot); font-family:var(--mono); font-size:21px; font-weight:700;
  letter-spacing:-.01em; word-break:break-all;}
.evi-path {margin:0 0 14px; color:var(--muted); font-size:15px;}
.evi-code {background:var(--panel); border:1px solid var(--rule); border-radius:var(--radius); padding:12px 0; overflow-x:auto; font-size:13px;}
.evi-code .code {padding:0; font-size:13px;}
.consulted {margin-top:26px; color:var(--muted); font-size:15px;}
.consulted ul {margin:10px 0 0; padding-left:18px; line-height:1.8;}
.sig {background:color-mix(in srgb,var(--ink) 9%,transparent);}
</style>
"""


DARK_ONLY = """
<style>
[data-testid="stFileUploaderDropzone"] {background:var(--field) !important; border:1px dashed var(--rule) !important;}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p {color:var(--muted) !important;}
[data-testid="stFileUploaderDropzone"] button {background:var(--panel) !important; color:var(--ink) !important;
  border:1px solid var(--rule) !important; border-radius:var(--radius) !important;}
[data-testid="stFileUploaderDropzone"] button p, [data-testid="stFileUploaderDropzone"] button span {color:var(--ink) !important;}
[data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] svg {color:var(--muted) !important; fill:var(--muted) !important; opacity:1 !important;}
[data-testid="stTooltipContent"], [data-baseweb="tooltip"] > div, [data-baseweb="tooltip"] [role="tooltip"] {background:var(--panel) !important;
  color:var(--ink) !important; border:1px solid var(--rule);}
[data-testid="stTooltipContent"] p {color:var(--ink) !important;}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileName"] {color:var(--ink) !important;}
</style>
"""


def theme_css(mode: str) -> str:
    vars_ = "".join(f"--{k}:{v};" for k, v in THEMES[mode].items())
    return CSS.replace("__VARS__", vars_) + (DARK_ONLY if mode == "dark" else "")


# ------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("theme", "light")
ss.setdefault("result", None)
ss.setdefault("sel", 0)
ss.setdefault("editing", True)
ss.setdefault("code_input", "")
ss.setdefault("mode", "Local folder")
ss.setdefault("folder", DEFAULT_REPO)
ss.setdefault("gh_url", "")
ss.setdefault("top_k", 5)
ss.setdefault("version", "v2")
ss.setdefault("compare", False)
ss.setdefault("nonce", 0)
ss.setdefault("n_funcs", None)
has_key = bool(os.environ.get("GROQ_API_KEY"))
ss.setdefault("use_mock", not has_key)
# Keep the settings alive across reruns. The widgets live in closed popovers, and Streamlit can
# otherwise forget their values when a rerun starts from another widget (for example the theme button).
for _k in ("code_input", "mode", "folder", "gh_url", "top_k", "version", "compare", "use_mock", "nonce"):
    ss[_k] = ss[_k]

st.markdown(theme_css(ss["theme"]), unsafe_allow_html=True)


# ----------------------------------------------------------------------- helpers
def esc(x) -> str:
    return html.escape(str(x))


GITHUB_RE = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?")


def fetch_github(url: str):
    """Shallow-clone (or download as zip) a PUBLIC GitHub repo. Returns (path, error)."""
    m = GITHUB_RE.fullmatch(url.strip())
    if not m:
        return None, "Enter a link like https://github.com/owner/repo"
    owner, name = m.groups()
    dest = os.path.join(CACHE_DIR, f"{owner}__{name}")
    if os.path.isdir(dest) and os.listdir(dest):
        return dest, None
    os.makedirs(CACHE_DIR, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}   # never wait for a password
    try:
        r = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "-q",
                            f"https://github.com/{owner}/{name}.git", dest],
                           capture_output=True, text=True, timeout=180, env=env)
        if r.returncode == 0:
            return dest, None
        err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "git clone failed"
        if "not found" in err.lower() or "authentication" in err.lower() or "could not read" in err.lower():
            return None, "Repository not found, or it is private. Only public repositories are supported."
        return None, f"Could not clone the repository: {err[:200]}"
    except FileNotFoundError:
        pass                                            # git not installed: fall back to a zip download
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        return None, "Cloning took too long (over 3 minutes). Try a smaller repository."
    try:
        zpath = dest + ".zip"
        urllib.request.urlretrieve(f"https://github.com/{owner}/{name}/archive/HEAD.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        os.remove(zpath)
        return dest, None
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(dest, ignore_errors=True)
        return None, f"Could not download the repository ({str(e)[:120]})."


@st.cache_resource(show_spinner=False)
def index_codebase(path: str, nonce: int):
    """Chunk and embed a codebase once; each folder gets its own Chroma collection."""
    chunks, n_files, skipped = collect_repo_chunks(path)
    total = len(chunks)
    if not chunks:
        return None, [], n_files, total
    chunks = chunks[:MAX_FUNCS]
    name = "cb_" + hashlib.md5(f"{os.path.abspath(path)}|{nonce}".encode()).hexdigest()[:16]
    collection = chromadb.Client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embed_texts([c["source"] for c in chunks]),
            documents=[c["source"] for c in chunks],
            metadatas=[{"function_name": c["function_name"], "file": c["file"], "line": c["line"],
                        "has_docstring": c["has_docstring"], "is_snake_case": c["is_snake_case"]} for c in chunks],
        )
    return collection, chunks, n_files, total


def make_prompt(version: str) -> ChatPromptTemplate:
    text = SYSTEM_PROMPT_V2 if version == "v2" else SYSTEM_PROMPT_V1
    text = text.replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages(
        [("system", text), ("human", "{references}\n\n### NEW FUNCTION TO REVIEW\n{new_code}")]
    )


def short(cid: str) -> str:
    path, _, name = _norm(cid).partition("::")
    return "/".join(path.split("/")[-2:]) + "::" + name


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


CATEGORY_ORDER = [
    ("function_naming", "Function naming"), ("param_naming", "Parameter naming"),
    ("vague_params", "Parameter clarity"), ("raises_exception", "Error handling"),
    ("bare_except", "Error handling"), ("logging_module", "Logging"),
    ("string_format", "String formatting"), ("import_star", "Imports"),
]


def categorize(c: dict):
    """Label a comment (keyword rules, same family as the evaluation scorer). Returns (key, label)."""
    text = re.sub("[\u2010-\u2015\u2212]", "-", f"{c.get('issue', '')} {c.get('comment', '')}".lower())
    for key, label in CATEGORY_ORDER:
        if CATEGORY_RULES[key](text):
            return key, label
    return "other", "Convention"


def lines_for(c: dict, code: str) -> list[int]:
    """Which lines of the reviewed code a comment is about (1-based). A display aid only: simple
    pattern rules on the code, no extra model call. Falls back to the first `def` line."""
    lines = code.splitlines()
    key, _ = categorize(c)
    text = f"{c.get('issue', '')} {c.get('comment', '')}".lower()
    patterns = {
        "raises_exception": r"^\s*raise\b",
        "bare_except": r"^\s*except\s*:",
        "logging_module": r"\blogging\.|\blogger\b|import logging",
        "string_format": r"\.format\(|%\s*[\(\w]",
        "import_star": r"import\s+\*",
    }
    pat = patterns.get(key)
    if pat is None and "print" in text:
        pat = r"\bprint\("
    hits = [i + 1 for i, ln in enumerate(lines) if pat and re.search(pat, ln)]
    if not hits:
        hits = [i + 1 for i, ln in enumerate(lines) if re.match(r"\s*(async\s+)?def\s", ln)][:1] or [1]
    return hits


TOKEN_RE = re.compile(r"(?P<comment>#.*$)|(?P<string>[fFrRbB]{0,2}\"[^\"]*\"|[fFrRbB]{0,2}'[^']*')"
                      r"|(?P<number>\b\d+(?:\.\d+)?\b)|(?P<word>[A-Za-z_]\w*)")


def highlight(line: str) -> str:
    out, pos, prev = [], 0, ""
    for m in TOKEN_RE.finditer(line):
        out.append(esc(line[pos:m.start()]))
        kind, tok = m.lastgroup, m.group()
        if kind == "word":
            if tok in keyword.kwlist:
                out.append(f'<span class="t-keyword">{esc(tok)}</span>')
            elif prev in ("def", "class"):
                out.append(f'<span class="t-defname">{esc(tok)}</span>')
            else:
                out.append(esc(tok))
            prev = tok
        else:
            out.append(f'<span class="t-{kind}">{esc(tok)}</span>')
            prev = ""
        pos = m.end()
    out.append(esc(line[pos:]))
    return "".join(out)


def code_html(code: str, marks: dict[int, list[int]] | None = None, sel: int = -1, first_line: int = 1) -> str:
    """Numbered, highlighted code. marks maps line number -> list of note indexes about that line."""
    marks = marks or {}
    rows = []
    for i, ln in enumerate(code.splitlines() or [""], 1):
        notes = marks.get(i, [])
        cls = "ln"
        if notes:
            cls += " marked" + (" sel" if sel in notes else "")
        foot = "".join(f'<span class="foot{"" if n == sel else " off"}">{chr(65 + n)}</span>' for n in notes)
        delay = f' style="animation-delay:{min(i, 30) * 12}ms"' if notes else ""
        rows.append(f'<div class="{cls}"{delay}><span class="no">{i + first_line - 1}</span>'
                    f'<span class="src">{highlight(ln) or " "}</span><span class="fn">{foot}</span></div>')
    return '<div class="code">' + "".join(rows) + "</div>"


def review_markdown(kept, repo_label, version):
    lines = ["# CodeLens review", f"Reviewed against: {repo_label} (prompt {version})", ""]
    if not kept:
        lines.append("No deviations found.")
    for i, c in enumerate(kept, 1):
        lines += [f"## {i}. {c.get('issue', 'Comment')}", c.get("comment", ""),
                  f"Based on: `{short(c.get('cited_source', ''))}`", ""]
    return "\n".join(lines)


SAMPLES = [("Bad function name", "test_cases/deviant_1.py"), ("Bare except", "test_cases/deviant_6.py"),
           ("Clean code", "test_cases/conformant_1.py")]
SAMPLES = [(lab, os.path.join(HERE, p)) for lab, p in SAMPLES if os.path.exists(os.path.join(HERE, p))]


def load_sample(path):
    ss["code_input"] = read(path)
    ss["editing"] = True
    ss["result"] = None


def pick(i):
    ss["sel"] = i


def toggle_theme():
    ss["theme"] = "dark" if ss["theme"] == "light" else "light"


def edit_code():
    ss["editing"] = True


def repo_name() -> str:
    if ss["mode"] == "GitHub repository" and ss["gh_url"].strip():
        m = GITHUB_RE.fullmatch(ss["gh_url"].strip())
        return m.group(2) if m else ss["gh_url"].strip()
    return os.path.basename(ss["folder"].rstrip("\\/")) or ss["folder"]


MARK = ('<svg width="26" height="26" viewBox="0 0 26 26" fill="none"><rect x="2" y="6" width="22" height="6" rx="2" '
        'fill="var(--gold)"/><circle cx="11" cy="12" r="6.5" stroke="currentColor" stroke-width="2.4"/>'
        '<path d="M16 17l6 6" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/></svg>')

# --------------------------------------------------------------------------- top bar
with st.container(key="topbar"):
    c_brand, c_note, c_repo, c_set, c_theme = st.columns([3.2, 3.2, 3.2, 1.2, 0.8], gap="small")
    c_brand.markdown(f'<div class="brand">{MARK}CodeLens</div>', unsafe_allow_html=True)
    if ss["use_mock"]:
        c_note.markdown('<div class="demo-note">Demo mode. No API key needed. The notes are not a real review.</div>',
                        unsafe_allow_html=True)
    meta = f" · {ss['n_funcs']} functions" if ss["n_funcs"] else ""
    with c_repo.popover(f"Reviewing against  {repo_name()}{meta}", use_container_width=True):
        st.markdown('<div style="font-family:var(--headf);font-weight:700;font-size:18px">Choose a codebase</div>',
                    unsafe_allow_html=True)
        st.radio("Codebase source", ["Local folder", "GitHub repository"], key="mode", horizontal=True,
                 label_visibility="collapsed")
        if ss["mode"] == "Local folder":
            st.text_input("Folder", key="folder", label_visibility="collapsed")
        else:
            st.text_input("GitHub URL", key="gh_url", placeholder="https://github.com/owner/repository",
                          label_visibility="collapsed")
            st.caption("Public repositories only. A shallow copy is kept in `repos_cache/`.")
        if st.button("Re-index this codebase", key="reindex"):
            ss["nonce"] += 1
            ss["n_funcs"] = None
    with c_set.popover("Settings", use_container_width=True):
        st.slider("Reference functions retrieved (top-k)", 3, 8, key="top_k")
        st.radio("Review prompt", ["v2", "v1"], key="version", horizontal=True,
                 help="v2 is the stricter prompt; v1 is the original.")
        st.checkbox("Demo mode (fake model, no API)", key="use_mock",
                    help="Tests the pipeline offline. The comments it shows are NOT real reviews.")
        st.checkbox("Also show the same model without your codebase", key="compare",
                    help="One extra API call, for comparison.")
        if not has_key and not ss["use_mock"]:
            st.error("GROQ_API_KEY is not set. Turn on demo mode, or set the key and restart.")
    c_theme.button("Dark" if ss["theme"] == "light" else "Light", key="theme_btn", on_click=toggle_theme,
                   use_container_width=True)

# ------------------------------------------------------------------------------ run
result = ss["result"]
status_slot = None


def run_review(code: str):
    """Index, retrieve, redact, review and verify. Returns (result dict, error string)."""
    if ss["mode"] == "GitHub repository":
        repo, err = fetch_github(ss["gh_url"])
    else:
        repo, err = ss["folder"], None
    if not err and not os.path.isdir(repo or ""):
        err = f"Folder not found: {repo}"
    if err:
        return None, err
    collection, chunks, n_files, total = index_codebase(repo, ss["nonce"])
    if not chunks:
        return None, "No Python functions were found in that codebase."
    ss["n_funcs"] = len(chunks)
    chain_mod.PROMPT = make_prompt(ss["version"])
    retriever = CodebaseRetriever(collection=collection, k=min(ss["top_k"], len(chunks)))
    llm = get_mock_llm() if ss["use_mock"] else get_llm()
    try:
        out = build_chain(retriever, llm).invoke({"new_code": code})
        generic = []
        if ss["compare"]:
            generic = build_baseline_chain(get_mock_baseline_llm() if ss["use_mock"] else llm).invoke({"new_code": code})
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            return None, "The model's rate or daily limit was reached. Wait and try again, or use demo mode."
        return None, f"The review didn't finish: {msg[:300]}"
    refs = {_norm(d.metadata["id"]): d for d in out["docs"]}
    return dict(code=code, kept=out["kept"], dropped=out["dropped"], refs=refs, generic=generic,
                redactions=out.get("redactions", []), version=ss["version"], mock=ss["use_mock"],
                label=repo if ss["mode"] == "Local folder" else ss["gh_url"].strip(), total=total,
                indexed=len(chunks)), None


PROGRESS = ["Reading your codebase", "Finding similar functions", "Hiding secrets and writing notes", "Checking every source"]


def progress_html(done: int) -> str:
    items = []
    for i, lab in enumerate(PROGRESS):
        cls = "done" if i < done else ("now" if i == done else "")
        dot = "&#10003;" if i < done else ""
        items.append(f'<li class="{cls}"><span class="dot">{dot}</span>{lab}</li>')
    return '<ul class="progress">' + "".join(items) + "</ul>"


# ---------------------------------------------------------------------------- columns
with st.container(key="cols"):
    col_code, col_notes, col_evi = st.columns([1.2, 1.0, 1.05], gap="small")

    # ---- column 1: your code -------------------------------------------------
    with col_code.container(key="col_code"):
        with st.container(key="head_code"):
            h1, h2 = st.columns([3, 2], vertical_alignment="center")
            h1.markdown('<div class="chead"><h2>Your code</h2></div>', unsafe_allow_html=True)
            with h2:
                if result is not None and not ss["editing"]:
                    st.button("Edit code", key="edit_btn", on_click=edit_code)
                    run_clicked = False
                else:
                    run_clicked = st.button("Review code", type="primary", key="run")
        if result is not None and not ss["editing"]:
            marks: dict[int, list[int]] = {}
            for n, c in enumerate(result["kept"]):
                for ln in lines_for(c, result["code"]):
                    marks.setdefault(ln, []).append(n)
            st.markdown(code_html(result["code"], marks, ss["sel"]), unsafe_allow_html=True)
        else:
            with st.container(key="samples"):
                st.markdown('<div class="lead" style="margin-bottom:14px;line-height:1.4">Paste your own code, or try a sample:</div>',
                            unsafe_allow_html=True)
                cols = st.columns(len(SAMPLES) + 2, vertical_alignment="center")
                for col, (lab, p) in zip(cols, SAMPLES):
                    col.button(lab, key=f"s_{lab}", on_click=load_sample, args=(p,))
                with cols[len(SAMPLES)].popover("Upload a file"):
                    up = st.file_uploader("Upload a Python file", type=["py"], label_visibility="collapsed")
                    if up is not None and ss.get("_uploaded") != up.file_id:
                        ss["_uploaded"] = up.file_id
                        ss["code_input"] = up.getvalue().decode("utf-8", errors="replace")
                        st.rerun()
            with st.container(key="code_box"):
                st.text_area("Python code", key="code_input", height=640, label_visibility="collapsed",
                             placeholder="Paste a function or a whole file here...")

    # ---- run (after the button, so the notes column can show progress) --------
    err = None
    if run_clicked:
        with col_notes.container(key="col_notes_run"):
            st.markdown('<div class="chead"><h2>Review notes</h2></div>', unsafe_allow_html=True)
            with st.container(key="run_body"):
                if not ss["code_input"].strip():
                    new, err = None, "Paste some code first, or pick a sample."
                else:
                    st.markdown(progress_html(1), unsafe_allow_html=True)
                    new, err = run_review(ss["code_input"])
                if err:
                    st.markdown(f'<div class="allclear" style="border-left-color:var(--accent)">'
                                f'<strong>The review didn\'t finish.</strong><p>{esc(err)}</p></div>', unsafe_allow_html=True)
        if new is not None:
            ss["result"], ss["sel"], ss["editing"] = new, 0, False
            st.rerun()

    # ---- column 2: review notes ---------------------------------------------
    if not run_clicked:
        with col_notes.container(key="col_notes"):
            with st.container(key="head_notes"):
                t1, t2 = st.columns([2, 2], vertical_alignment="center")
                t1.markdown('<div class="chead"><h2>Review notes</h2></div>', unsafe_allow_html=True)
                with t2:
                    if result is not None:
                        st.download_button("Export review", review_markdown(result["kept"], result["label"], result["version"]),
                                           file_name="codelens_review.md", mime="text/markdown", key="export")
            with st.container(key="notes_body"):
                if err:
                    st.markdown(f'<div class="allclear" style="border-left-color:var(--accent)"><strong>The review didn\'t finish.</strong>'
                                f'<p>{esc(err)}</p></div>', unsafe_allow_html=True)
                elif result is None:
                    st.markdown('<div class="empty"><div class="lede">Review code the way a teammate who knows the project would.</div>'
                                '<p>Paste a function, or try a sample. Every note points to a real function in your codebase, '
                                'so you can check where it came from.</p>'
                                '<p>CodeLens drops any note that cites a function it did not actually retrieve.</p></div>',
                                unsafe_allow_html=True)
                else:
                    kept, dropped = result["kept"], result["dropped"]
                    n_ref = len(result["refs"])
                    st.markdown(f'<div class="summary">{len(kept)} note{"" if len(kept) == 1 else "s"}, compared with '
                                f'{n_ref} function{"" if n_ref == 1 else "s"} from <code>{esc(repo_name())}</code>. '
                                f'{len(result["redactions"])} secret{"" if len(result["redactions"]) == 1 else "s"} hidden '
                                f'before the model saw your code.</div>', unsafe_allow_html=True)
                    if not kept:
                        st.markdown('<div class="allclear"><strong>Nothing to change.</strong>'
                                    '<p>Your code matches the conventions of the functions it was compared with.</p></div>',
                                    unsafe_allow_html=True)
                    for i, c in enumerate(kept):
                        _, cat = categorize(c)
                        sel = " sel" if i == ss["sel"] else ""
                        st.markdown(
                            f'<div class="note{sel}"><span class="letter">{chr(65 + i)}</span><div class="nbody">'
                            f'<div class="ntitle">{esc(c.get("issue", "Comment"))}<span class="cat">{esc(cat)}</span></div>'
                            f'<div class="ntext">{esc(c.get("comment", ""))}</div>'
                            f'<div class="nsrc">based on <code>{esc(short(c.get("cited_source", ""))).replace("::", "::<wbr>")}</code></div></div></div>',
                            unsafe_allow_html=True)
                        with st.container(key=f"pick_{i}"):
                            if i != ss["sel"]:
                                st.button("Show this one", key=f"pickbtn_{i}", on_click=pick, args=(i,))
                    if result["generic"] is not None and ss["compare"]:
                        items = "".join(f'<li><strong>{esc(g.get("issue", "Comment"))}</strong>. {esc(g.get("comment", ""))}</li>'
                                        for g in result["generic"])
                        st.markdown('<div class="generic"><h3>Without your codebase</h3>'
                                    '<div class="muted">The same model, given only your code. None of these cite a source.</div>'
                                    f'<ul>{items or "<li>No comments.</li>"}</ul></div>', unsafe_allow_html=True)
                    if dropped:
                        rows = "".join(f'<p>{esc(d.get("comment") or d.get("error", ""))}<br><code>cited: {esc(d.get("cited_source", ""))}</code></p>'
                                       for d in dropped)
                        st.markdown(f'<details class="dropped"><summary>{len(dropped)} note{"" if len(dropped) == 1 else "s"} '
                                    f'dropped: cited a function that was not retrieved</summary>{rows}</details>',
                                    unsafe_allow_html=True)
                    if result["redactions"]:
                        rows = "".join(f'<li>{esc(r["type"])}, line {esc(r["line"])}: <code>{esc(r["preview"])}</code></li>'
                                       for r in result["redactions"])
                        st.markdown(f'<details class="dropped"><summary>Secrets hidden before the model saw anything '
                                    f'({len(result["redactions"])})</summary><ul>{rows}</ul></details>', unsafe_allow_html=True)

    # ---- column 3: evidence ---------------------------------------------------
    with col_evi.container(key="col_evi"):
        with st.container(key="head_evi"):
            st.markdown('<div class="chead"><h2>From your codebase</h2></div>', unsafe_allow_html=True)
        with st.container(key="evi_body"):
            if result is None or not result["kept"]:
                msg = ("Nothing to show: no note cites a function." if result is not None else
                       "Pick a note and the function it is based on appears here, next to your code.")
                st.markdown(f'<div class="empty"><p>{msg}</p></div>', unsafe_allow_html=True)
            else:
                sel = min(ss["sel"], len(result["kept"]) - 1)
                c = result["kept"][sel]
                cid = _norm(c.get("cited_source", ""))
                d = result["refs"].get(cid)
                path, _, fname = cid.partition("::")
                st.markdown(f'<p class="evi-for">Note {chr(65 + sel)} is based on</p><p class="evi-name">{esc(fname)}</p>'
                            f'<p class="evi-path">{esc("/".join(path.split("/")[-2:]))}'
                            f'{" · line " + str(d.metadata.get("line")) if d is not None and d.metadata.get("line") else ""}</p>',
                            unsafe_allow_html=True)
                if d is not None:
                    first = int(d.metadata.get("line") or 1)
                    st.markdown(f'<div class="evi-code">{code_html(d.page_content, first_line=first)}</div>', unsafe_allow_html=True)
                lis = "".join(f'<li><code>{esc(short(k))}</code></li>' for k in result["refs"])
                st.markdown(f'<details class="consulted"><summary>Functions consulted ({len(result["refs"])})</summary>'
                            f'<ul>{lis}</ul></details>', unsafe_allow_html=True)
                model = "mock" if result["mock"] else os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
                st.markdown(f'<p class="muted" style="margin-top:18px">Prompt {result["version"]} · top-k {len(result["refs"])} · model {esc(model)}</p>',
                            unsafe_allow_html=True)