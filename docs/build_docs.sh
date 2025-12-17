#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-}"
JUPYTER_BOOK_BIN="${JUPYTER_BOOK:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Unable to find a python interpreter. Set the PYTHON env var." >&2
    exit 1
  fi
fi

if [[ -z "$JUPYTER_BOOK_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/jupyter-book" ]]; then
    JUPYTER_BOOK_BIN="$REPO_ROOT/.venv/bin/jupyter-book"
  elif command -v jupyter-book >/dev/null 2>&1; then
    JUPYTER_BOOK_BIN="$(command -v jupyter-book)"
  else
    echo "Unable to find jupyter-book. Install it or set the JUPYTER_BOOK env var." >&2
    exit 1
  fi
fi

JUPYTER_BOOK_VERSION="$(
  "$PYTHON_BIN" -c 'import importlib.metadata as m; print(m.version("jupyter-book"))' 2>/dev/null || true
)"
if [[ "$JUPYTER_BOOK_VERSION" =~ ^2(\.|$) ]]; then
  cat >&2 <<'EOF'
Detected Jupyter Book v2, but this repository uses the classic Jupyter Book
configuration (_config.yml/_toc.yml). Install a v0.x release instead:

  python -m pip install 'jupyter-book<2'
EOF
  exit 1
fi

"$PYTHON_BIN" "$REPO_ROOT/docs/prepare_docs_assets.py"
"$JUPYTER_BOOK_BIN" build "$REPO_ROOT/docs"
