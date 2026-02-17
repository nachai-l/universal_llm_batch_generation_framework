"""
HTML Rendering Utilities (Pipeline 6)

Intent
- Deterministic Markdown → HTML rendering
- No external dependencies
- Stable output for reproducible reports
"""

from __future__ import annotations

import html
from typing import List


# -----------------------------
# Minimal Markdown → HTML
# -----------------------------

def _md_to_html_basic(md_text: str) -> str:
    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    lines = md_text.splitlines()
    out: List[str] = []

    i = 0
    in_ul = False

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def is_table_row(line: str) -> bool:
        s = line.strip()
        return s.startswith("|") and s.endswith("|") and "|" in s[1:-1]

    def is_table_sep(line: str) -> bool:
        s = line.strip()
        if not is_table_row(s):
            return False
        parts = [p.strip() for p in s[1:-1].split("|")]
        for p in parts:
            if not p:
                return False
            if set(p) <= set("-:") and "-" in p:
                continue
            return False
        return True

    def parse_table_row(line: str) -> List[str]:
        return [p.strip() for p in line.strip()[1:-1].split("|")]

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if not s:
            close_ul()
            i += 1
            continue

        # Headings
        if s.startswith("### "):
            close_ul()
            out.append(f"<h3>{esc(s[4:])}</h3>")
            i += 1
            continue
        if s.startswith("## "):
            close_ul()
            out.append(f"<h2>{esc(s[3:])}</h2>")
            i += 1
            continue
        if s.startswith("# "):
            close_ul()
            out.append(f"<h1>{esc(s[2:])}</h1>")
            i += 1
            continue

        # Tables
        if is_table_row(s) and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            close_ul()
            header = parse_table_row(s)
            i += 2
            rows: List[List[str]] = []
            while i < len(lines) and is_table_row(lines[i]):
                rows.append(parse_table_row(lines[i]))
                i += 1

            out.append("<div class='table-wrap'>")
            out.append("<table>")
            out.append("<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in header) + "</tr></thead>")
            out.append("<tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{esc(c)}</th>" for c in r) + "</tr>")
            out.append("</tbody></table></div>")
            continue

        # Bullet list
        if s.startswith("- "):
            if not in_ul:
                close_ul()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{esc(s[2:])}</li>")
            i += 1
            continue

        # Paragraph
        close_ul()
        out.append(f"<p>{esc(s)}</p>")
        i += 1

    close_ul()
    return "\n".join(out)


# -----------------------------
# Public Renderer
# -----------------------------

def render_report_html(*, md_text: str, title: str = "Pipeline 6 Report") -> str:
    body_html = _md_to_html_basic(md_text)

    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "<meta charset=\"utf-8\" />\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"<title>{html.escape(title)}</title>\n"
        "<style>\n"
        ":root{--bg:#0b0f14;--card:#111827;--text:#e5e7eb;--muted:#9ca3af;--line:#243244;--th:#0f172a;}\n"
        "body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui;}\n"
        ".container{max-width:980px;margin:0 auto;padding:28px 16px 56px;}\n"
        ".card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;}\n"
        "h1{font-size:26px;} h2{font-size:18px;margin-top:20px;} h3{font-size:15px;margin-top:14px;}\n"
        "p{margin:8px 0;} ul{margin:8px 0 8px 20px;} li{margin:6px 0;}\n"
        ".table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;margin:10px 0 14px;}\n"
        "table{border-collapse:collapse;width:100%;min-width:520px;}\n"
        "th,td{padding:10px 12px;border-bottom:1px solid var(--line);}\n"
        "th{background:var(--th);text-align:left;font-weight:600;}\n"
        "tr:last-child td{border-bottom:none;}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<div class=\"container\">\n"
        "<div class=\"card\">\n"
        f"{body_html}\n"
        "</div>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
