import subprocess
import autogen_code_review_bot.pr_analysis as pr


def test_timeout_returns_message(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get('args', []), timeout=1)

    monkeypatch.setattr(pr, 'run', fake_run)
    out = pr._run_command(['cmd'], cwd='.', timeout=1)
    assert out == 'timed out'
