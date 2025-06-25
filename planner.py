import json
import pathlib
import re


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    plan_path = pathlib.Path("DEVELOPMENT_PLAN.md")
    if not plan_path.exists():
        raise SystemExit("DEVELOPMENT_PLAN.md not found")
    content = plan_path.read_text()

    match = re.search(r"^- \[ \] \*\*Feature:\*\* (.+)", content, re.MULTILINE)
    epic = match.group(1).strip() if match else None
    if not epic:
        raise SystemExit("No pending epics found")

    tasks = []
    criteria = {}

    if "Multi-Language Support" in epic:
        tasks = [
            {"task": "Add language detection utility", "priority": "P1"},
            {
                "task": "Integrate language-specific linters in analyze_pr",
                "priority": "P1",
            },
            {"task": "Load linter configuration from YAML", "priority": "P2"},
            {"task": "Document multi-language setup in README", "priority": "P3"},
        ]
        criteria = {
            "add-language-detection-utility": {
                "description": "Add language detection utility",
                "cases": {
                    "detects_known_extensions": "Files with .py, .js, .ts, .go, .rs return correct language",
                    "unknown_extension": "Returns 'unknown' for unrecognized extensions",
                },
            },
            "integrate-language-specific-linters-in-analyze-pr": {
                "description": "Integrate language-specific linters in analyze_pr",
                "cases": {
                    "python_linter": "Runs ruff for Python files",
                    "js_linter_missing": "Outputs 'not installed' if eslint is missing",
                },
            },
            "load-linter-configuration-from-yaml": {
                "description": "Load linter configuration from YAML",
                "cases": {
                    "yaml_parsed": "Reads languages and linters from config file",
                    "fallback_defaults": "Uses default tools when not specified",
                },
            },
            "document-multi-language-setup-in-readme": {
                "description": "Document multi-language setup in README",
                "cases": {
                    "instructions_present": "README explains how to enable extra languages",
                    "examples_included": "Provides sample config snippet",
                },
            },
        }
    else:
        slug = slugify(epic)
        tasks = [{"task": f"Implement {epic}", "priority": "P1"}]
        criteria = {
            slug: {
                "description": f"Implement {epic}",
                "cases": {
                    "default_case": "Placeholder for acceptance criteria",
                },
            }
        }

    board_lines = [
        "# Sprint Board",
        "",
        "## Backlog",
        "| Task | Owner | Priority | Status |",
        "| --- | --- | --- | --- |",
    ]
    for item in tasks:
        board_lines.append(f"| {item['task']} | @agent | {item['priority']} | Todo |")

    pathlib.Path("SPRINT_BOARD.md").write_text("\n".join(board_lines) + "\n")

    for key, val in criteria.items():
        file_slug = key.replace("-", "_")
        val["test_file"] = f"tests/{file_slug}.py"

    pathlib.Path("tests/sprint_acceptance_criteria.json").write_text(
        json.dumps(criteria, indent=2)
    )
    print("Sprint board and acceptance criteria generated for epic:", epic)


if __name__ == "__main__":
    main()
