from autogen_code_review_bot.agents import load_agents_from_yaml, run_dual_review
from autogen_code_review_bot import analyze_and_comment
import yaml


def test_load_agents_from_yaml(tmp_path):
    cfg = {
        'agents': {
            'coder': {'model': 'gpt-4', 'temperature': 0.2, 'focus_areas': ['bugs']},
            'reviewer': {'model': 'gpt-4', 'focus_areas': ['security']},
        }
    }
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.dump(cfg))
    agents = load_agents_from_yaml(str(path))
    assert set(agents.keys()) == {'coder', 'reviewer'}
    assert agents['coder'].config.temperature == 0.2


def test_run_dual_review(tmp_path):
    cfg = {
        'agents': {
            'coder': {'model': 'gpt-4'},
            'reviewer': {'model': 'gpt-4'},
        }
    }
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.dump(cfg))
    feedback = run_dual_review('print("hi")', str(path))
    assert 'coder' in feedback
    assert 'reviewer' in feedback


def test_public_api_exposes_analyze_and_comment():
    assert callable(analyze_and_comment)
