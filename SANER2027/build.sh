#!/usr/bin/env bash
# Build the SANER 2027 submission PDF (pdflatex -> bibtex -> pdflatex x2).
# Usage: ./build.sh            (from anywhere)
set -euo pipefail
cd "$(dirname "$0")"
MAIN=SANER2027-main-labeling
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error "$MAIN.tex" >/dev/null
bibtex "$MAIN" >/dev/null
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error "$MAIN.tex" >/dev/null
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error "$MAIN.tex" >/dev/null
echo "== $MAIN.pdf: $(pdfinfo "$MAIN.pdf" | awk '/^Pages/{print $2}') pages"
echo "== Overfull boxes (should be none):"
grep -E "^Overfull" "$MAIN.log" || echo "   none"
echo "== Undefined references / citations:"
grep -E "Warning.*(undefined|Undefined)" "$MAIN.log" || echo "   none"
