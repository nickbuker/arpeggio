# AGENTS.md

## Project Context
This is a Python-based project using sequential transaction data to predict fraud.

## Development Commands
- **Setup:** `uv sync`
- **Run Agent:** `uv run python -m src.main`
- **Linting:** `uv run ruff check . --fix`
- **Testing:** `uv run pytest tests/`

## Coding Standards
- **Style:** Follow [NumPy docstring style](https://discuss.scientific-python.org/t/agents-md-and-claude-md-addition-to-scipy-repository/2233) for all public functions.
- **Formatting:** Use Black-compatible formatting via Ruff.
- **Type Hints:** All function signatures MUST include Python type hints.
- **Testing Requirements:** 
  - Every new feature must include unit tests in `tests/`.
  - Use the Arrange-Act-Assert pattern for test structure.

## Boundaries & Constraints
- ✅ **Always:** Use `pydantic.BaseModel` for data validation.
- ⚠️ **Ask first:** Before adding new top-level dependencies to `pyproject.toml`.
- 🚫 **Never:** Commit `.env` files or hardcode API keys.
- 🚫 **Never:** Use `pip` directly; always use `uv` commands.
