#!/usr/bin/env python3
"""Page-budget check for the SANER 2027 submission (10 pages of main text + at most 2 of references).

Run after SANER2027/build.sh:
    python3 docs/reframe/pagecheck.py [--final] [path/to/pdf]

Reports where the main text ends (the last line before the REFERENCES heading, i.e. the end of the
Data Availability statement), how many body-text lines are still free on page 10, where the reference
list starts and ends, and how much room is left on page 12.

The user's target for the final draft: the main text fills exactly 10 pages (page 10 full, at most
FULL_TOLERANCE lines free) and the references start at the top of page 11 (a \\clearpage before the
bibliography guarantees the page break). During intermediate sessions the free lines on page 10 are
the budget for new prose; with --final the check fails unless page 10 is full.
Exit code 1 if a limit (or, with --final, the fill target) is broken.
"""
from __future__ import annotations

import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

MAIN_LIMIT = 10      # last page that may hold main text
TOTAL_LIMIT = 12     # last page that may hold references
FULL_TOLERANCE = 3   # --final: at most this many free body lines at the bottom of page 10

LINE_RE = re.compile(r'<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>', re.S)
WORD_RE = re.compile(r'<word[^>]*>(.*?)</word>', re.S)
PAGE_RE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', re.S)


def load(pdf: Path):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bbox.html"
        subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(out)], check=True)
        html = out.read_text(encoding="utf-8", errors="replace")
    pages = []
    for pno, m in enumerate(PAGE_RE.finditer(html), start=1):
        width = float(m.group(1))
        lines = []
        for lm in LINE_RE.finditer(m.group(3)):
            x0, y0, x1, y1 = map(float, lm.group(1, 2, 3, 4))
            text = " ".join(WORD_RE.findall(lm.group(5)))
            col = 0 if x0 < width / 2 else 1
            lines.append(dict(page=pno, col=col, x0=x0, y0=y0, x1=x1, y1=y1, text=text))
        lines.sort(key=lambda l: (l["col"], l["y0"]))
        pages.append(dict(width=width, lines=lines))
    return pages


def body_metrics(pages):
    """Column top/bottom and body line height, measured on pages 2..9."""
    bottoms, tops, gaps = [], [], []
    for p in pages[1:9]:
        for col in (0, 1):
            ls = [l for l in p["lines"] if l["col"] == col and 6 < l["y1"] - l["y0"] < 12.5]
            if not ls:
                continue
            bottoms.append(max(l["y1"] for l in ls))
            tops.append(min(l["y0"] for l in ls))
            ys = sorted(l["y0"] for l in ls)
            gaps += [b - a for a, b in zip(ys, ys[1:]) if 11.0 < b - a < 13.0]
    lh = statistics.median(gaps) if gaps else 11.955
    bottom = sorted(bottoms)[int(0.9 * (len(bottoms) - 1))]
    top = statistics.median(tops)
    return top, bottom, lh


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--final"]
    final = "--final" in sys.argv[1:]
    pdf = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "SANER2027" / "SANER2027-main-labeling.pdf"
    pages = load(pdf)
    top, bottom, lh = body_metrics(pages)
    col_lines = (bottom - top) / lh
    order = [l for p in pages for l in p["lines"]]

    ref_idx = next((i for i, l in enumerate(order) if l["text"].replace(" ", "").upper() == "REFERENCES"), None)
    if ref_idx is None:
        print("Could not find the REFERENCES heading.")
        return 2
    end = order[ref_idx - 1]
    main_slack = (bottom - end["y1"]) / lh
    if end["col"] == 0:
        main_slack += col_lines
    main_slack += 2 * col_lines * (MAIN_LIMIT - end["page"])

    last_page = len(pages)
    last_lines = pages[-1]["lines"]
    ref_bottom = max(l["y1"] for l in last_lines)
    # \balance splits the last page evenly between the two columns, so free space counts twice.
    ref_free_pts = 2 * (bottom - ref_bottom) + 2 * (bottom - top) * (TOTAL_LIMIT - last_page)
    ref_slack = ref_free_pts / lh

    ref_head = order[ref_idx]
    print(f"PDF: {pdf}  ({len(pages)} pages)")
    print(f"Main text ends: page {end['page']}, {'left' if end['col'] == 0 else 'right'} column, "
          f"y={end['y1']:.0f} of {bottom:.0f}  (last words: '...{end['text'][-40:]}')")
    print(f"  free lines before the end of page {MAIN_LIMIT}: {main_slack:.0f} body lines")
    print(f"References start: page {ref_head['page']}, {'left' if ref_head['col'] == 0 else 'right'} column, "
          f"y={ref_head['y0']:.0f}")
    print(f"References end: page {last_page}, lowest line y={ref_bottom:.0f} of {bottom:.0f}")
    print(f"  room left before the page-{TOTAL_LIMIT} limit: about {ref_slack:.0f} body-line equivalents "
          f"(one new reference entry takes about 2-3)")

    ok = end["page"] <= MAIN_LIMIT and last_page <= TOTAL_LIMIT
    if not ok:
        print("OVER THE LIMIT: fix before committing.")
        return 1
    print("OK: within 10 + 2 pages.")
    full = end["page"] == MAIN_LIMIT and main_slack <= FULL_TOLERANCE and ref_head["page"] == MAIN_LIMIT + 1
    if full:
        print(f"FINAL TARGET MET: page {MAIN_LIMIT} is full and the references start on page {MAIN_LIMIT + 1}.")
    else:
        print(f"Final target not met yet: page {MAIN_LIMIT} must be full (<= {FULL_TOLERANCE} free lines) "
              f"and the references must start on page {MAIN_LIMIT + 1}.")
        if final:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
