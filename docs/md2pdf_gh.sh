#!/usr/bin/env bash
# Convert a Markdown file to a GitHub-styled PDF via pandoc + WeasyPrint.
# - CJK (Chinese) via system fonts (Heiti SC / PingFang SC)
# - GitHub-flavored light theme (docs/github-md.css)
# - Pygments syntax highlighting
# - Clickable links
#
# Usage: bash docs/md2pdf_gh.sh docs/<file>.md
set -euo pipefail

md="${1:-}"
if [ -z "$md" ] || [ ! -f "$md" ]; then
  echo "usage: $0 <file.md>" >&2
  exit 1
fi

out="${md%.md}.pdf"
title="$(basename "${md%.md}")"
css="$(cd "$(dirname "$0")" && pwd)/github-md.css"

pandoc "$md" \
  -f gfm \
  -o "$out" \
  --pdf-engine=weasyprint \
  --css="$css" \
  --metadata title="$title" \
  --toc --toc-depth=3 \
  --highlight-style=pygments \
  --standalone

echo "wrote $out"
