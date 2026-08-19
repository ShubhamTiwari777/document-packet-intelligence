"""Render the Markdown technical report to a paginated PDF.

Uses PyMuPDF's Story engine (HTML + a CSS subset -> laid-out pages) rather than adding a heavy
PDF toolchain: PyMuPDF is already a dependency for page parsing, so the report builds with no
extra system packages and no headless browser.
"""
from __future__ import annotations

import argparse
import html as html_lib
import re
from pathlib import Path

import markdown
import pymupdf

CSS = """
body { font-family: sans-serif; font-size: 8.6pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 16pt; font-weight: bold; margin: 0 0 4pt 0; color: #10203a; }
h2 { font-size: 11.5pt; font-weight: bold; margin: 11pt 0 3pt 0; color: #10203a; }
h3 { font-size: 9.6pt; font-weight: bold; margin: 8pt 0 2pt 0; color: #1d3557; }
h4 { font-size: 8.8pt; font-weight: bold; margin: 7pt 0 2pt 0; color: #2b4c7e; }
p  { margin: 0 0 4pt 0; }
ul, ol { margin: 0 0 4pt 0; padding-left: 11pt; }
li { margin: 0 0 1.5pt 0; }
code { font-family: monospace; font-size: 7.8pt; color: #7a2518; }
pre { font-family: monospace; font-size: 7.2pt; background: #f4f5f7; padding: 4pt;
      margin: 0 0 5pt 0; line-height: 1.25; }
table { width: 100%; margin: 0 0 6pt 0; border-collapse: collapse; }
th { font-size: 7.8pt; text-align: left; background: #eef1f6; font-weight: bold;
     padding: 2.5pt 3pt; border-bottom: 1px solid #b8c2d0; color: #10203a; }
td { font-size: 7.8pt; padding: 2.2pt 3pt; border-bottom: 1px solid #e2e6ec; }
blockquote { margin: 0 0 5pt 6pt; padding-left: 6pt; border-left: 2px solid #c0692b; color: #444; }
hr { margin: 7pt 0; }
a { color: #1d4f91; }
"""


def markdown_to_html(source: str) -> str:
    # Story does not implement the `---` horizontal rule reliably; drop the separators, the
    # heading hierarchy already carries the structure.
    source = re.sub(r"^---\s*$", "", source, flags=re.MULTILINE)
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    # Story's CSS subset has no `white-space: pre`, so code blocks arrive as one long line.
    # Convert newlines inside <pre> to explicit breaks.
    def fix_pre(match: re.Match) -> str:
        inner = match.group(1)
        return "<pre>" + inner.replace("\n", "<br/>") + "</pre>"
    body = re.sub(r"<pre>(.*?)</pre>", fix_pre, body, flags=re.DOTALL)
    body = body.replace("<code><br/>", "<code>").replace("</code>", "</code>")
    return f"<html><head><style>{CSS}</style></head><body>{body}</body></html>"


def build(markdown_path: Path, output_path: Path, margin: float = 40.0) -> int:
    source = markdown_path.read_text(encoding="utf-8")
    story = pymupdf.Story(html=markdown_to_html(source), user_css=None)
    writer = pymupdf.DocumentWriter(str(output_path))
    page_rect = pymupdf.paper_rect("a4")
    content = page_rect + (margin, margin, -margin, -margin)

    pages = 0
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content)
        story.draw(device)
        writer.end_page()
        pages += 1
        if pages > 60:  # runaway guard
            break
    writer.close()
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="technical_report.md")
    parser.add_argument("--output", default="technical_report.pdf")
    args = parser.parse_args()
    count = build(Path(args.input), Path(args.output))
    size = Path(args.output).stat().st_size
    print(f"Wrote {args.output}: {count} pages, {size/1024:.0f} KB")
