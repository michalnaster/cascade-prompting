"""Render the article to a print-ready PDF.

Markdown -> styled HTML -> Chrome headless --print-to-pdf. No LaTeX toolchain
required, and the equation/chart PNGs are picked up as ordinary images.

Run with:  uv run --with markdown python scripts/make_pdf.py
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "cascade-prompting-token-latency-efficiency.md"
OUT = ROOT / "cascade-prompting-token-latency-efficiency.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font: 10.5pt/1.62 Charter, "Bitstream Charter", Georgia, "Times New Roman", serif;
  color: #16150f; background: #fff; margin: 0; hyphens: auto;
}
h1 {
  font-size: 25pt; line-height: 1.16; letter-spacing: -0.015em;
  margin: 0 0 6mm; font-weight: 700;
}
h2 {
  font-size: 15pt; margin: 11mm 0 3.5mm; padding-top: 3mm;
  border-top: 1.2px solid #d9d8d1; break-after: avoid; font-weight: 700;
}
h3 { font-size: 12pt; margin: 8mm 0 2.5mm; break-after: avoid; font-weight: 700; }
p { margin: 0 0 3.4mm; text-align: justify; }
h1 + p { /* deck */
  font-size: 11.5pt; font-style: italic; color: #4a4941;
  border-left: 2.5px solid #2a78d6; padding-left: 4mm; text-align: left;
  margin-bottom: 6mm;
}
strong { font-weight: 700; }
a { color: #1f5fa8; text-decoration: none; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.6pt;
  background: #f2f1ec; padding: 0.6mm 1.1mm; border-radius: 2px;
}
pre {
  background: #f7f6f1; border: 1px solid #e3e2dd; border-radius: 3px;
  padding: 3.5mm 4mm; overflow: hidden; break-inside: avoid; margin: 0 0 4mm;
}
pre code { background: none; padding: 0; font-size: 8.2pt; line-height: 1.45; }
blockquote {
  margin: 0 0 4.5mm; padding: 3.5mm 5mm; background: #f7f6f1;
  border-left: 2.5px solid #b8b7b1; break-inside: avoid;
}
blockquote p { margin-bottom: 2.6mm; text-align: left; }
blockquote p:last-child { margin-bottom: 0; }
blockquote table { margin: 3mm 0; }
ul, ol { margin: 0 0 3.4mm; padding-left: 6mm; }
li { margin-bottom: 2.2mm; text-align: justify; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 5mm;
  font-size: 8.6pt; break-inside: avoid; font-variant-numeric: tabular-nums;
}
th, td { padding: 1.7mm 2.4mm; text-align: left; vertical-align: top; }
thead th { border-bottom: 1.2px solid #16150f; font-weight: 700; }
tbody tr { border-bottom: 0.6px solid #e3e2dd; }
tbody tr:last-child { border-bottom: 1.2px solid #16150f; }
td code, th code { font-size: 8pt; background: none; padding: 0; }
img { display: block; max-width: 100%; margin: 5mm auto; break-inside: avoid; }
img[src*="/eq"] { max-width: 78%; margin: 4mm auto 5mm; }
hr { border: none; border-top: 1px solid #d9d8d1; margin: 7mm 0; }
"""


def main() -> int:
    if not pathlib.Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}")

    text = SRC.read_text()
    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list", "smarty"]
    )
    # Absolute file:// paths so Chrome resolves the figures from a temp dir.
    html_body = re.sub(
        r'src="(figures/[^"]+)"', lambda m: f'src="{(ROOT / m.group(1)).as_uri()}"', html_body
    )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{SRC.stem}</title><style>{CSS}</style></head>"
        f"<body>{html_body}</body></html>"
    )

    with tempfile.TemporaryDirectory() as tmp:
        html_path = pathlib.Path(tmp) / "article.html"
        html_path.write_text(page)
        pdf_path = pathlib.Path(tmp) / "article.pdf"
        subprocess.run(
            [
                CHROME, "--headless=new", "--disable-gpu",
                "--no-pdf-header-footer",
                "--allow-file-access-from-files",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        shutil.copy(pdf_path, OUT)

    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
