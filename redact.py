"""Secret redaction: runs on code BEFORE it is sent to the LLM (Unit IV: responsible AI).

Code often contains API keys, passwords and tokens. CodeLens sends code to an external
LLM, so secrets are replaced with <REDACTED:type> first. Pure Python, no API needed.

    from redact import redact
    clean, findings = redact(source_code)
"""
import math
import re

# (type, regex). Order matters: specific key formats first, generic assignment last.
PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("url_password", re.compile(r"(?<=://)[^\s:/@]+:([^\s@/]{3,})(?=@)")),
]

# name = "value"  where the name looks secret-like
ASSIGN = re.compile(
    r"""(?P<name>\b[\w.]*(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|auth|credential)[\w]*)"""
    r"""(?P<sep>\s*[:=]\s*)(?P<q>["'])(?P<val>[^"'\n]{6,})(?P=q)""",
    re.I,
)
PLACEHOLDER = re.compile(r"^(your[_-]?|<|\{|\$\{|xxx|changeme|example|test|dummy|none|null|\*+)", re.I)


def shannon(s: str) -> float:
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


def _mask(s: str) -> str:
    return (s[:2] + "..." + s[-2:]) if len(s) > 8 else "***"


def redact(text: str):
    """Return (redacted_text, findings). findings: list of {type, line, preview}."""
    findings = []

    def line_of(pos, src):
        return src.count("\n", 0, pos) + 1

    for kind, rx in PATTERNS:
        src = text

        def sub(m, kind=kind, src=src):
            secret = m.group(1) if kind == "url_password" else m.group(0)
            findings.append({"type": kind, "line": line_of(m.start(), src), "preview": _mask(secret)})
            if kind == "url_password":
                return m.group(0).replace(secret, f"<REDACTED:{kind}>")
            return f"<REDACTED:{kind}>"

        text = rx.sub(sub, text)

    src = text

    def assign_sub(m):
        val = m.group("val")
        if PLACEHOLDER.match(val) or val.startswith("<REDACTED"):
            return m.group(0)
        # a secret-looking NAME with a real-looking value (not a short word like "bearer")
        if len(val) < 8 and shannon(val) < 2.5:
            return m.group(0)
        findings.append({"type": "secret_assignment", "line": line_of(m.start(), src), "preview": _mask(val)})
        return f"{m.group('name')}{m.group('sep')}{m.group('q')}<REDACTED:secret_assignment>{m.group('q')}"

    text = ASSIGN.sub(assign_sub, text)
    findings.sort(key=lambda f: f["line"])
    return text, findings


if __name__ == "__main__":
    import sys

    code = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    clean, found = redact(code)
    print(clean)
    print(f"\n{len(found)} secret(s) redacted:")
    for f in found:
        print(f"  line {f['line']}: {f['type']} ({f['preview']})")
