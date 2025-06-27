import requests
from autogen_code_review_bot.github_integration import get_pull_request_diff, post_comment

def test_get_pull_request_diff_calls_api(monkeypatch):
    called = {}

    def fake_get(url, headers, params, **kwargs):
        called['url'] = url
        called['headers'] = headers
        called['params'] = params
        class Resp:
            text = 'diff'
            def raise_for_status(self):
                pass
        return Resp()

    monkeypatch.setattr(requests, 'get', fake_get)
    diff = get_pull_request_diff('owner/repo', 42, 'token')
    assert '/repos/owner/repo/pulls/42' in called['url']
    assert diff == 'diff'


def test_post_comment_calls_api(monkeypatch):
    called = {}
    def fake_post(url, headers, data, **kwargs):
        called['url'] = url
        called['data'] = data
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {'ok': True}
        return Resp()
    monkeypatch.setattr(requests, 'post', fake_post)
    resp = post_comment('owner/repo', 42, 'hi', 'token')
    assert '/repos/owner/repo/issues/42/comments' in called['url']
    assert resp == {'ok': True}
