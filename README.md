# autogen-code-review-bot

A minimal, working two-agent code review loop in Python.

**Author** writes code → **Reviewer** critiques → iterate until it passes (or hit max iterations).

No frameworks required. Swap in any LLM backend.

---

## How it works

```
Task description
      │
      ▼
┌─────────────┐     code      ┌──────────────┐
│ AuthorAgent │ ──────────►  │ ReviewerAgent │
│             │ ◄──────────  │               │
└─────────────┘   feedback   └──────────────┘
      │                              │
      │  (if FAIL and iterations     │
      │   < max)                     │
      └──────────── revise ──────────┘
```

The Reviewer returns structured feedback in four categories:
- **correctness** — bugs, wrong logic
- **style** — PEP 8, naming, docs
- **edge_cases** — None, empty input, boundaries
- **security** — injection, unsafe eval, unsafe defaults

The loop continues until the Reviewer passes all categories or `max_iterations` is hit.

---

## Quick start

```bash
pip install -e .

# Run the mock demo (no API key needed)
python examples/demo_mock.py

# Run with real Anthropic API
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo_anthropic.py
```

---

## Usage

```python
from autogen_code_review_bot import run_review_loop
from autogen_code_review_bot.backends import MockBackend, AnthropicBackend

# With mock (good for testing / CI)
result = run_review_loop(
    "Write a function that validates email addresses",
    author_backend=MockBackend(responses=["def validate_email(e): ..."]),
    reviewer_backend=MockBackend(responses=["STATUS: pass\n..."]),
)

# With real Anthropic backend
result = run_review_loop(
    "Write a function that validates email addresses",
    backend=AnthropicBackend(model="claude-3-5-haiku-20241022"),
    max_iterations=5,
    verbose=True,
)

print(result.passed)       # True / False
print(result.final_code)   # The final produced code
print(result.summary())    # Human-readable iteration history
```

### Author and Reviewer can use different backends

```python
result = run_review_loop(
    task,
    author_backend=AnthropicBackend(model="claude-3-5-haiku-20241022"),
    reviewer_backend=AnthropicBackend(model="claude-opus-4-5"),
)
```

---

## API

### `run_review_loop(task, *, backend=None, author_backend=None, reviewer_backend=None, max_iterations=5, verbose=False) → ReviewResult`

| Parameter | Description |
|-----------|-------------|
| `task` | Natural-language coding task |
| `backend` | Shared backend for both agents (shortcut) |
| `author_backend` | Backend for AuthorAgent only |
| `reviewer_backend` | Backend for ReviewerAgent only |
| `max_iterations` | Stop after this many cycles (default 5) |
| `verbose` | Print progress to stdout |

### `ReviewResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `passed` | `bool` | Did the reviewer pass the final code? |
| `final_code` | `str` | The last produced code |
| `final_feedback` | `ReviewFeedback` | Last reviewer output |
| `iterations` | `List[IterationRecord]` | Full history |
| `iteration_count` | `int` | How many cycles ran |
| `.summary()` | `str` | Human-readable report |

### Backends

| Class | Description |
|-------|-------------|
| `MockBackend(responses=[...])` | Scripted responses, no API call |
| `MockBackend(response_fn=fn)` | Dynamic mock via callable |
| `AnthropicBackend(model=...)` | Real Anthropic API (requires `anthropic` package) |

Custom backend: subclass `Backend` and implement `complete(system, messages) -> str`.

---

## Running tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

---

## Project structure

```
src/autogen_code_review_bot/
├── __init__.py      # Public API
├── backends.py      # MockBackend, AnthropicBackend, Backend ABC
├── agents.py        # AuthorAgent, ReviewerAgent + parsing helpers
├── loop.py          # run_review_loop orchestrator
└── models.py        # ReviewFeedback, ReviewComment, ReviewStatus

tests/
├── test_models.py
├── test_backends.py
├── test_agents.py
└── test_loop.py

examples/
├── demo_mock.py      # No API key needed
└── demo_anthropic.py # Real Anthropic API
```

---

## License

MIT
