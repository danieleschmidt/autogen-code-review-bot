import yaml
from autogen_code_review_bot.pr_analysis import load_linter_config


def test_yaml_parsed(tmp_path):
    cfg = {"linters": {"python": "mypy", "javascript": "eslint"}}
    path = tmp_path / "linters.yaml"
    path.write_text(yaml.dump(cfg))
    config = load_linter_config(str(path))
    assert config["python"] == "mypy"
    assert config["javascript"] == "eslint"
    assert config["typescript"] == "eslint"  # from defaults
    assert config["ruby"] == "rubocop"  # new default


def test_fallback_defaults(tmp_path):
    cfg = {"linters": {"python": "flake8"}}
    path = tmp_path / "linters.yaml"
    path.write_text(yaml.dump(cfg))
    config = load_linter_config(str(path))
    # specified language overridden
    assert config["python"] == "flake8"
    # unspecified languages fall back
    assert config["javascript"] == "eslint"
    assert config["ruby"] == "rubocop"

