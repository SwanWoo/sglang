#!/usr/bin/env bash
# Convert a Markdown file to PDF via pandoc + xelatex.
# - CJK (Chinese) rendered with Heiti SC
# - Latin + symbols (arrows, stars, check) with Arial Unicode MS (broad glyph coverage)
# - Code blocks with Menlo
# - Clickable hyperlinks via hyperref (colorlinks)
#
# Usage: bash docs/md2pdf.sh docs/<file>.md
set -euo pipefail

md="${1:-}"
if [ -z "$md" ] || [ ! -f "$md" ]; then
  echo "usage: $0 <file.md>" >&2
  exit 1
fi

out="${md%.md}.pdf"
title="$(basename "${md%.md}")"

pandoc "$md" \
  -f gfm \
  -o "$out" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Heiti SC" \
  -V CJKmonofont="Heiti SC" \
  -V mainfont="Arial Unicode MS" \
  -V monofont="Menlo" \
  -V geometry:margin=2cm \
  -V colorlinks=true \
  -V linkcolor=NavyBlue \
  -V urlcolor=NavyBlue \
  -V linktoc=all \
  --metadata title="$title" \
  --toc --toc-depth=3 \
  --highlight-style=tango

echo "wrote $out"
