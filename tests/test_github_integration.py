import requests
from autogen_code_review_bot.github_integration import (
    get_pull_request_diff,
    post_comment,
)

def test_get_pull_request_diff_calls_api(monkeypatch):
    called = {}

    def fake_request(method, url, headers=None, params=None, data=None, timeout=10):
        called['method'] = method
        called['url'] = url
        called['headers'] = headers
        called['params'] = params
        class Resp:
            text = 'diff'
            def raise_for_status(self):
                pass
        return Resp()
    monkeypatch.setattr(requests, 'request', fake_request)
    diff = get_pull_request_diff('owner/repo', 42, 'token')
    assert called['method'] == 'get'
    assert '/repos/owner/repo/pulls/42' in called['url']
    assert diff == 'diff'


def test_post_comment_calls_api(monkeypatch):
    called = {}
    def fake_request(method, url, headers=None, data=None, params=None, timeout=10):
        called['method'] = method
        called['url'] = url
        called['data'] = data
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {'ok': True}
        return Resp()
    monkeypatch.setattr(requests, 'request', fake_request)
    resp = post_comment('owner/repo', 42, 'hi', 'token')
    assert called['method'] == 'post'
    assert '/repos/owner/repo/issues/42/comments' in called['url']
    assert resp == {'ok': True}
