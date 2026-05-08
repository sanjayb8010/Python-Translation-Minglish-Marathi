# Contributing Guide

Thanks for contributing to this project.

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Copy environment template:

```bash
copy .env.example .env
```

4. Fill required Azure values in `.env`.
5. Run locally:

```bash
python app.py
```

## Branch Naming

Use descriptive branch names:

- `feature/<short-name>`
- `fix/<short-name>`
- `chore/<short-name>`

## Commit Messages

Use concise, purpose-oriented messages:

- `feat: add ...`
- `fix: resolve ...`
- `docs: update ...`
- `chore: ...`

## Pull Requests

Before opening a PR:

- Ensure app starts and health endpoint works
- Verify changed behavior manually
- Update docs when behavior/config changes
- Keep PRs focused and reviewable

PRs should include:

- What changed
- Why it changed
- How to test
- Any risks or follow-up tasks

## Security Requirements

- Never commit `.env`, credentials, or secrets
- Rotate credentials immediately if leaked
- Do not include real keys in screenshots or logs

## Code Style

- Keep changes minimal and targeted
- Prefer clear naming over clever logic
- Add brief comments only where non-obvious behavior exists
