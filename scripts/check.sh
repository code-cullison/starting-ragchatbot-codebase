#!/bin/bash
# Run all quality checks (formatting check + tests). Exits non-zero on failure.
set -e
cd "$(dirname "$0")/.."

echo "==> black --check"
uv run black --check backend main.py

echo "==> pytest"
(cd backend && uv run pytest -q)

echo "All quality checks passed."
