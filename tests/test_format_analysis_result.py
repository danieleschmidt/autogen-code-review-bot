from autogen_code_review_bot.github_integration import format_analysis_result
from autogen_code_review_bot.pr_analysis import PRAnalysisResult, AnalysisSection

def test_format_analysis_result():
    result = PRAnalysisResult(
        security=AnalysisSection(tool="bandit", output="sec"),
        style=AnalysisSection(tool="ruff", output="style"),
        performance=AnalysisSection(tool="radon", output="perf"),
    )
    body = format_analysis_result(result)
    assert body.startswith("## 🤖 AutoGen Code Review")
    assert "Security (bandit)" in body
    assert "Performance (radon)" in body
