from autogen_code_review_bot.pr_analysis import analyze_pr


def test_python_linter(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    result = analyze_pr(str(tmp_path))
    assert "ruff" in result.style.tool


def test_js_linter_missing(tmp_path, monkeypatch):
    (tmp_path / "app.js").write_text("console.log('hi');\n")
    monkeypatch.setenv("PATH", "")
    result = analyze_pr(str(tmp_path))
    assert "not installed" in result.style.output

