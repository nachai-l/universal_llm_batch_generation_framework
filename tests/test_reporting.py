# tests/test_reporting.py

from __future__ import annotations

from functions.core.reporting import (
    ReportStats,
    compute_report_stats,
    render_report_md,
)


def _make_record(*, role="Engineer", qset="QS1", qid="Q1", qtype="behavioral",
                 qname="Tell me about yourself", judge=None):
    rec = {
        "input": {
            "Role Track Example Name": role,
            "Question Set": qset,
            "Question ID": qid,
            "Question Type": qtype,
        },
        "meta": {"group_key": role},
        "parsed": {"question_name": qname},
    }
    if judge is not None:
        rec["judge"] = judge
    return rec


def test_compute_report_stats_basic_counts():
    records = [
        _make_record(role="A", qset="S1", qtype="t1"),
        _make_record(role="A", qset="S2", qtype="t1"),
        _make_record(role="B", qset="S1", qtype="t2"),
    ]
    stats = compute_report_stats(
        records=records,
        sample_per_role=0,
        include_full_examples=False,
        judge_enabled_hint=False,
        max_reason_examples=5,
    )
    assert isinstance(stats, ReportStats)
    assert stats.n_records == 3
    assert stats.groups.by_role == {"A": 2, "B": 1}
    assert stats.groups.by_set == {"S1": 2, "S2": 1}
    assert stats.groups.by_type == {"t1": 2, "t2": 1}


def test_compute_report_stats_judge_aggregation():
    records = [
        _make_record(judge={"verdict": "PASS", "score": 8, "reasons": ["good"]}),
        _make_record(judge={"verdict": "FAIL", "score": 3, "reasons": ["bad", "poor"]}),
        _make_record(judge={"verdict": "PASS", "score": 9, "reasons": ["good"]}),
    ]
    stats = compute_report_stats(
        records=records,
        sample_per_role=0,
        include_full_examples=False,
        judge_enabled_hint=True,
        max_reason_examples=5,
    )
    assert stats.judge.enabled is True
    assert stats.judge.n_with_judge == 3
    assert stats.judge.n_pass == 2
    assert stats.judge.n_fail == 1
    assert stats.judge.score_min == 3
    assert stats.judge.score_max == 9
    assert stats.judge.score_avg is not None
    # top reasons: "good" appears 2x, "bad" 1x, "poor" 1x
    reason_dict = dict(stats.judge.top_reasons)
    assert reason_dict["good"] == 2


def test_compute_report_stats_samples():
    records = [
        _make_record(role="A", qname="Q one"),
        _make_record(role="A", qname="Q two"),
        _make_record(role="A", qname="Q three"),
        _make_record(role="B", qname="Q four"),
    ]
    stats = compute_report_stats(
        records=records,
        sample_per_role=2,
        include_full_examples=False,
        judge_enabled_hint=False,
        max_reason_examples=5,
    )
    assert len(stats.sample_groups["A"]) == 2
    assert len(stats.sample_groups["B"]) == 1


def test_compute_report_stats_empty():
    stats = compute_report_stats(
        records=[],
        sample_per_role=0,
        include_full_examples=False,
        judge_enabled_hint=False,
        max_reason_examples=5,
    )
    assert stats.n_records == 0
    assert stats.judge.n_with_judge == 0


def test_compute_report_stats_missing_input_and_meta():
    records = [{"parsed": {"question_name": "x"}}]
    stats = compute_report_stats(
        records=records,
        sample_per_role=0,
        include_full_examples=False,
        judge_enabled_hint=False,
        max_reason_examples=5,
    )
    assert stats.n_records == 1
    assert stats.data_quality["missing_input_count"] == 1
    assert stats.data_quality["missing_meta_count"] == 1


def test_render_report_md_contains_sections():
    records = [
        _make_record(role="Dev", qset="S1", qtype="tech",
                     judge={"verdict": "PASS", "score": 7, "reasons": ["ok"]}),
    ]
    stats = compute_report_stats(
        records=records,
        sample_per_role=1,
        include_full_examples=False,
        judge_enabled_hint=True,
        max_reason_examples=5,
    )
    md = render_report_md(
        meta={"generated_at_utc": "2026-01-01T00:00:00Z", "parameters_path": "test.yaml"},
        stats=stats,
        sample_per_group=1,
        include_full_examples=False,
    )
    assert "# Pipeline 6" in md
    assert "## Run Summary" in md
    assert "## Counts by Role" in md
    assert "## Judge" in md
    assert "## Samples" in md
    assert "## Data Quality" in md


def test_render_report_md_judge_disabled():
    stats = compute_report_stats(
        records=[_make_record()],
        sample_per_role=0,
        include_full_examples=False,
        judge_enabled_hint=False,
        max_reason_examples=5,
    )
    md = render_report_md(
        meta={"generated_at_utc": "now", "parameters_path": "p.yaml"},
        stats=stats,
        sample_per_group=0,
        include_full_examples=False,
    )
    assert "_Judge disabled._" in md
    assert "_Sampling disabled._" in md
