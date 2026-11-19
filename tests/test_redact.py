import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from redact import redact

def kinds(code):
    return [f["type"] for f in redact(code)[1]]

def test_known_key_formats():
    code = 'a="AKIAIOSFODNN7EXAMPLE"\nb="gsk_' + "a1B2c3D4" * 4 + '"\nc="ghp_' + "x" * 36 + '"\n'
    clean, found = redact(code)
    assert "AKIAIOSFODNN7EXAMPLE" not in clean
    assert "gsk_a1B2" not in clean and "ghp_xxxx" not in clean
    assert {"aws_access_key", "groq_key", "github_token"} <= set(kinds(code))

def test_assignment_with_secret_name():
    clean, found = redact('db_password = "Tr0ub4dor&3xyz"\n')
    assert "Tr0ub4dor" not in clean and "db_password" in clean and len(found) == 1

def test_placeholders_and_normal_code_untouched():
    code = 'password = "your_password_here"\ntoken = None\nwindow = 21\nname = "volatility"\n'
    clean, found = redact(code)
    assert clean == code and found == []

def test_private_key_block_and_url_password():
    code = "k='''-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----'''\nu='postgres://admin:hunter22@db/x'\n"
    clean, found = redact(code)
    assert "MIIabc" not in clean and "hunter22" not in clean
    assert {"private_key", "url_password"} <= set(kinds(code))

def test_line_numbers_and_no_secret_in_findings():
    code = "x = 1\ny = 2\napi_key = 'sk-" + "Z9" * 15 + "'\n"
    _, found = redact(code)
    assert found[0]["line"] == 3
    assert "Z9Z9Z9Z9" not in str(found)

def test_idempotent():
    code = 'secret = "abcdEFGH1234!!"\n'
    once, _ = redact(code)
    twice, f2 = redact(once)
    assert once == twice and f2 == []

def test_real_repo_style_code_has_no_false_positives():
    code = "def calculate_ewma_volatility(returns, span=21):\n    token_count = len(returns)\n    return returns.ewm(span=span).std()\n"
    assert redact(code) == (code, [])
