from __future__ import annotations

from functions.io.html_rendering import render_report_html


def test_render_report_html_wraps_document() -> None:
    md = "# Title\n\nHello\n"
    html = render_report_html(md_text=md, title="My Report")

    assert html.startswith("<!doctype html>")
    assert "<html>" in html
    assert "<head>" in html
    assert "<body>" in html
    assert "<div class=\"container\">" in html
    assert "<div class=\"card\">" in html
    assert "<title>My Report</title>" in html


def test_render_report_html_escapes_title() -> None:
    html = render_report_html(md_text="# X\n", title='A&B "X"<Y>')
    # title is escaped
    assert "<title>A&amp;B &quot;X&quot;&lt;Y&gt;</title>" in html


def test_md_headings_render() -> None:
    md = "# H1\n\n## H2\n\n### H3\n"
    html = render_report_html(md_text=md)

    assert "<h1>H1</h1>" in html
    assert "<h2>H2</h2>" in html
    assert "<h3>H3</h3>" in html


def test_md_paragraph_render_and_escape() -> None:
    md = "Hello <world> & \"friends\"\n"
    html = render_report_html(md_text=md)

    # rendered as paragraph and escaped
    assert "<p>Hello &lt;world&gt; &amp; &quot;friends&quot;</p>" in html


def test_md_bullets_render_as_ul() -> None:
    md = "- a\n- b\n\n- c\n"
    html = render_report_html(md_text=md)

    # should produce two UL blocks (one for a,b then another for c)
    assert html.count("<ul>") == 2
    assert html.count("</ul>") == 2
    assert "<li>a</li>" in html
    assert "<li>b</li>" in html
    assert "<li>c</li>" in html


def test_md_table_renders_table_with_header_and_rows() -> None:
    md = (
        "## Table\n\n"
        "| Key | Value |\n"
        "|---|---|\n"
        "| a | 1 |\n"
        "| b | 2 |\n"
    )
    html = render_report_html(md_text=md)

    assert "<div class='table-wrap'>" in html
    assert "<table>" in html
    assert "<thead><tr><th>Key</th><th>Value</th></tr></thead>" in html
    assert "<tbody>" in html
    assert "<tr><td>a</th><td>1</th></tr>" in html
    assert "<tr><td>b</th><td>2</th></tr>" in html
    assert "</table>" in html


def test_md_table_does_not_trigger_without_separator() -> None:
    # looks like a table row but missing separator row => should be paragraphs, not a table
    md = "| Key | Value |\n| a | 1 |\n"
    html = render_report_html(md_text=md)

    assert "<table>" not in html
    # will be treated as paragraphs (escaped pipes remain)
    assert "<p>| Key | Value |</p>" in html
    assert "<p>| a | 1 |</p>" in html


def test_md_table_separator_allows_colons() -> None:
    md = (
        "| Key | Value |\n"
        "|:---|---:|\n"
        "| a | 1 |\n"
    )
    html = render_report_html(md_text=md)
    assert "<table>" in html
    assert "<th>Key</th>" in html
    assert "<th>Value</th>" in html
    assert "<tr><td>a</th><td>1</th></tr>" in html


def test_html_escapes_markdown_content_in_table_cells_and_list_items() -> None:
    md = (
        "- <x>\n\n"
        "| A | B |\n"
        "|---|---|\n"
        "| <x> | & |\n"
    )
    html = render_report_html(md_text=md)

    # list item escaped
    assert "<li>&lt;x&gt;</li>" in html
    # table cells escaped
    assert "<tr><td>&lt;x&gt;</th><td>&amp;</th></tr>" in html
