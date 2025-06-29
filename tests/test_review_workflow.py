import requests
from autogen_code_review_bot.github_integration import (
    _request_with_retries,
    analyze_and_comment,
)
from autogen_code_review_bot.pr_analysis import PRAnalysisResult, AnalysisSection


def test_request_retries(monkeypatch):
    calls = {'count': 0}

    def fake_request(method, url, headers=None, data=None, params=None, timeout=10):
        calls['count'] += 1
        if calls['count'] < 2:
            raise requests.RequestException('boom')

        class Resp:
            def raise_for_status(self):
                pass

            text = 'ok'

        return Resp()

    monkeypatch.setattr(requests, 'request', fake_request)
    resp = _request_with_retries('get', 'http://x', token='t')
    assert resp.text == 'ok'
    assert calls['count'] == 2


def test_analyze_and_comment(monkeypatch):
    result = PRAnalysisResult(
        security=AnalysisSection('bandit', 'sec'),
        style=AnalysisSection('ruff', 'style'),
        performance=AnalysisSection('radon', 'perf'),
    )

    def fake_analyze(repo_path, config_path=None):
        return result

    posted = {}

    def fake_post_comment(repo, pr, body, token):
        posted['body'] = body
        return {'ok': True}

    monkeypatch.setattr('autogen_code_review_bot.pr_analysis.analyze_pr', fake_analyze)
    monkeypatch.setattr('autogen_code_review_bot.github_integration.post_comment', fake_post_comment)
    resp = analyze_and_comment('repo_path', 'owner/r', 1, 't')
    assert resp == {'ok': True}
    assert 'Security' in posted['body']
    assert 'Style' in posted['body']
    assert 'Performance' in posted['body']
