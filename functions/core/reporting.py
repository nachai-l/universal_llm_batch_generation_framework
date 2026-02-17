# functions/core/reporting.py
"""
Reporting (Pipeline 6)

Intent
- Pure logic: compute summary stats from exported JSONL and manifests
- Render deterministic report markdown (+ optional HTML)

No I/O here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class JudgeStats:
    enabled: bool
    n_with_judge: int
    n_pass: int
    n_fail: int
    score_min: Optional[int]
    score_avg: Optional[float]
    score_max: Optional[int]
    top_reasons: List[Tuple[str, int]]  # (reason, count)


@dataclass(frozen=True)
class GroupCounts:
    by_role: Dict[str, int]
    by_set: Dict[str, int]
    by_type: Dict[str, int]
    by_role_set: Dict[str, Dict[str, int]]   # role -> set -> count
    by_role_type: Dict[str, Dict[str, int]]  # role -> type -> count


@dataclass(frozen=True)
class SampleItem:
    role: str
    question_set: str
    question_id: str
    question_type: str
    question_name: str
    # Optional full content
    example_answer_good: Optional[str] = None
    example_answer_mid: Optional[str] = None
    example_answer_bad: Optional[str] = None
    grading_rubrics: Optional[str] = None


@dataclass(frozen=True)
class ReportStats:
    n_records: int
    judge: JudgeStats
    groups: GroupCounts
    sample_groups: Dict[str, List[SampleItem]]
    data_quality: Dict[str, Any]


# -----------------------------
# Small helpers
# -----------------------------

def _inc(m: Dict[str, int], k: str, n: int = 1) -> None:
    m[k] = int(m.get(k, 0)) + int(n)


def _inc2(m: Dict[str, Dict[str, int]], k1: str, k2: str, n: int = 1) -> None:
    if k1 not in m:
        m[k1] = {}
    _inc(m[k1], k2, n=n)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _stable_sort_dict(d: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}


def _stable_sort_nested(d: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, inner in sorted(d.items(), key=lambda kv: str(kv[0])):
        out[str(k)] = _stable_sort_dict(inner)
    return out


def _get_role(input_obj: Dict[str, Any], meta_obj: Dict[str, Any]) -> str:
    # Prefer input Role Track Example Name (classic interview-prep CSV)
    role = _safe_str(input_obj.get("Role Track Example Name")).strip()
    if role:
        return role
    # Fallback to pipeline meta
    role2 = _safe_str(meta_obj.get("group_key")).strip()
    if role2:
        return role2
    return "(missing role)"


def _get_qset_qid_qtype(input_obj: Dict[str, Any]) -> Tuple[str, str, str]:
    qset = _safe_str(input_obj.get("Question Set")).strip()
    qid = _safe_str(input_obj.get("Question ID")).strip()
    qtype = _safe_str(input_obj.get("Question Type")).strip()
    return qset, qid, qtype


def _push_sample(
    *,
    samples_by_role: Dict[str, List[SampleItem]],
    role: str,
    si: SampleItem,
    sample_per_role: int,
) -> None:
    if sample_per_role <= 0:
        return
    cur = samples_by_role.get(role, [])
    if len(cur) >= sample_per_role:
        return
    cur.append(si)
    samples_by_role[role] = cur


# -----------------------------
# Compute stats
# -----------------------------

def compute_report_stats(
    *,
    records: Iterable[Dict[str, Any]],
    sample_per_role: int,
    include_full_examples: bool,
    judge_enabled_hint: bool,
    max_reason_examples: int,
) -> ReportStats:
    # Group counts
    by_role: Dict[str, int] = {}
    by_set: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_role_set: Dict[str, Dict[str, int]] = {}
    by_role_type: Dict[str, Dict[str, int]] = {}

    # Judge stats
    n_with_judge = 0
    n_pass = 0
    n_fail = 0
    score_sum = 0
    score_n = 0
    score_min: Optional[int] = None
    score_max: Optional[int] = None
    reason_counts: Dict[str, int] = {}

    # Samples
    samples_by_role: Dict[str, List[SampleItem]] = {}

    # Data quality
    dq_missing_keys: Dict[str, int] = {}
    dq_invalid_judge = 0
    dq_missing_input = 0
    dq_missing_meta = 0
    dq_group_output_records = 0

    n_records = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue

        n_records += 1

        input_obj = rec.get("input")
        if not isinstance(input_obj, dict):
            dq_missing_input += 1
            input_obj = {}

        meta_obj = rec.get("meta")
        if not isinstance(meta_obj, dict):
            dq_missing_meta += 1
            meta_obj = {}

        role = _get_role(input_obj, meta_obj)
        qset, qid, qtype = _get_qset_qid_qtype(input_obj)

        _inc(by_role, role)
        _inc(by_set, qset or "(missing set)")
        _inc(by_type, qtype or "(missing type)")
        _inc2(by_role_set, role, qset or "(missing set)")
        _inc2(by_role_type, role, qtype or "(missing type)")

        parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}
        judge = rec.get("judge") if isinstance(rec.get("judge"), dict) else None

        # Basic key presence checks (sanity)
        for k in ("parsed", "meta", "input"):
            if k not in rec:
                _inc(dq_missing_keys, k)

        # Judge stats
        if judge is not None:
            n_with_judge += 1
            verdict = _safe_str(judge.get("verdict")).strip().upper()
            if verdict == "PASS":
                n_pass += 1
            elif verdict == "FAIL":
                n_fail += 1
            else:
                dq_invalid_judge += 1

            sc = judge.get("score")
            if isinstance(sc, int):
                score_sum += sc
                score_n += 1
                score_min = sc if score_min is None else min(score_min, sc)
                score_max = sc if score_max is None else max(score_max, sc)

            reasons = judge.get("reasons")
            if isinstance(reasons, list):
                for r in reasons:
                    rs = _safe_str(r).strip()
                    if rs:
                        _inc(reason_counts, rs)

        # Samples
        # Support both:
        #  - Row-output: parsed has question_name + answers fields
        #  - Group-output: parsed has "questions": [ ... ]
        if sample_per_role > 0:
            questions = parsed.get("questions")
            if isinstance(questions, list) and questions:
                dq_group_output_records += 1
                # pick the first question as the "sample" for the group record
                q0 = questions[0] if isinstance(questions[0], dict) else {}
                si = SampleItem(
                    role=role,
                    question_set=_safe_str(q0.get("question_set") or qset),
                    question_id=_safe_str(q0.get("question_id") or qid),
                    question_type=_safe_str(q0.get("question_type") or qtype),
                    question_name=_safe_str(q0.get("question_name")),
                    example_answer_good=_safe_str(q0.get("example_answer_good")) if include_full_examples else None,
                    example_answer_mid=_safe_str(q0.get("example_answer_mid")) if include_full_examples else None,
                    example_answer_bad=_safe_str(q0.get("example_answer_bad")) if include_full_examples else None,
                    grading_rubrics=_safe_str(q0.get("grading_rubrics")) if include_full_examples else None,
                )
                _push_sample(samples_by_role=samples_by_role, role=role, si=si, sample_per_role=sample_per_role)
            else:
                # row-output style
                si = SampleItem(
                    role=role,
                    question_set=qset,
                    question_id=qid,
                    question_type=qtype,
                    question_name=_safe_str(parsed.get("question_name")),
                    example_answer_good=_safe_str(parsed.get("example_answer_good")) if include_full_examples else None,
                    example_answer_mid=_safe_str(parsed.get("example_answer_mid")) if include_full_examples else None,
                    example_answer_bad=_safe_str(parsed.get("example_answer_bad")) if include_full_examples else None,
                    grading_rubrics=_safe_str(parsed.get("grading_rubrics")) if include_full_examples else None,
                )
                _push_sample(samples_by_role=samples_by_role, role=role, si=si, sample_per_role=sample_per_role)

    score_avg = (score_sum / score_n) if score_n > 0 else None

    # Top reasons (deterministic): count desc then reason asc
    top_reasons = sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))
    if max_reason_examples is not None and int(max_reason_examples) >= 0:
        top_reasons = top_reasons[: int(max_reason_examples)]

    judge_stats = JudgeStats(
        enabled=bool(judge_enabled_hint) or (n_with_judge > 0),
        n_with_judge=int(n_with_judge),
        n_pass=int(n_pass),
        n_fail=int(n_fail),
        score_min=score_min,
        score_avg=round(float(score_avg), 4) if score_avg is not None else None,
        score_max=score_max,
        top_reasons=[(str(r), int(c)) for r, c in top_reasons],
    )

    group_counts = GroupCounts(
        by_role=_stable_sort_dict(by_role),
        by_set=_stable_sort_dict(by_set),
        by_type=_stable_sort_dict(by_type),
        by_role_set=_stable_sort_nested(by_role_set),
        by_role_type=_stable_sort_nested(by_role_type),
    )

    dq = {
        "missing_top_level_keys": _stable_sort_dict(dq_missing_keys),
        "missing_input_count": int(dq_missing_input),
        "missing_meta_count": int(dq_missing_meta),
        "invalid_judge_count": int(dq_invalid_judge),
        "group_output_records_detected": int(dq_group_output_records),
    }

    return ReportStats(
        n_records=int(n_records),
        judge=judge_stats,
        groups=group_counts,
        sample_groups={k: v[:] for k, v in sorted(samples_by_role.items(), key=lambda x: x[0])},
        data_quality=dq,
    )


# -----------------------------
# Markdown rendering helpers
# -----------------------------

def _md_escape_cell(v: Any) -> str:
    """
    Escape markdown table cells: pipes + newlines.
    Keep it minimal and deterministic.
    """
    s = "" if v is None else str(v)
    s = s.replace("\n", "<br>")
    s = s.replace("|", "\\|")
    return s


def _md_table_kv(title: str, d: Dict[str, Any]) -> str:
    lines = [f"## {title}", "", "| Key | Value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {_md_escape_cell(k)} | {_md_escape_cell(v)} |")
    lines.append("")
    return "\n".join(lines)


def _md_table_counts(title: str, d: Dict[str, int]) -> str:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {_md_escape_cell(k)} | {int(v)} |")
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Markdown renderer (Pipeline 6)
# -----------------------------

def render_report_md(
    *,
    meta: Dict[str, Any],
    stats: ReportStats,
    sample_per_group: int,
    include_full_examples: bool,
    max_reason_examples: int = 5,
) -> str:
    out: List[str] = []

    out.append("# Pipeline 6 — Report")
    out.append("")
    out.append(f"- Generated at (UTC): {meta.get('generated_at_utc')}")
    out.append(f"- Parameters: `{meta.get('parameters_path')}`")
    out.append("")

    # Run Summary (stable)
    run_summary = {
        "n_records": stats.n_records,
        "judge_enabled": stats.judge.enabled,
        "judge_pass": stats.judge.n_pass,
        "judge_fail": stats.judge.n_fail,
        "judge_score_min": stats.judge.score_min,
        "judge_score_avg": stats.judge.score_avg,
        "judge_score_max": stats.judge.score_max,
        "output_jsonl": meta.get("output_jsonl"),
        "output_psv": meta.get("output_psv"),
        "pipeline4_manifest": meta.get("pipeline4_manifest_path"),
        "pipeline5_manifest": meta.get("pipeline5_manifest_path"),
    }
    run_summary = {k: v for k, v in run_summary.items() if v is not None}
    out.append(_md_table_kv("Run Summary", run_summary))

    # Counts
    out.append(_md_table_counts("Counts by Role", stats.groups.by_role))
    out.append(_md_table_counts("Counts by Question Set", stats.groups.by_set))
    out.append(_md_table_counts("Counts by Question Type", stats.groups.by_type))

    # Judge Top Reasons
    out.append("## Judge")
    out.append("")
    top_reasons = stats.judge.top_reasons[: max(0, int(max_reason_examples))]
    if not stats.judge.enabled:
        out.append("_Judge disabled._")
        out.append("")
    elif not top_reasons:
        out.append("_No judge reasons found._")
        out.append("")
    else:
        out.append("### Top Reasons")
        out.append("")
        out.append("| Reason | Count |")
        out.append("|---|---|")
        for r, c in top_reasons:
            out.append(f"| {_md_escape_cell(r)} | {int(c)} |")
        out.append("")

    # Samples
    out.append("## Samples")
    out.append("")
    out.append(f"- sample_per_group: {int(sample_per_group)}")
    out.append(f"- include_full_examples: {bool(include_full_examples)}")
    out.append("")

    if sample_per_group <= 0:
        out.append("_Sampling disabled._")
        out.append("")
    elif not stats.sample_groups:
        out.append("_No samples available._")
        out.append("")
    else:
        for role, items in sorted(stats.sample_groups.items(), key=lambda x: x[0]):
            out.append(f"### {role}")
            out.append("")
            for si in items:
                prefix_parts: List[str] = []
                if si.question_set or si.question_id:
                    if si.question_set and si.question_id:
                        prefix_parts.append(f"({si.question_set}/{si.question_id})")
                    else:
                        prefix_parts.append(f"({si.question_set or si.question_id})")
                if si.question_type:
                    prefix_parts.append(f"[{si.question_type}]")

                prefix = " ".join(prefix_parts).strip()
                title = si.question_name.strip() if si.question_name.strip() else "(sample)"
                out.append(f"- {prefix} {title}".strip())

                if include_full_examples:
                    out.append("")
                    if si.example_answer_good is not None:
                        out.append("**Example Answer – Good**")
                        out.append(_md_escape_cell(si.example_answer_good))
                        out.append("")
                    if si.example_answer_mid is not None:
                        out.append("**Example Answer – Mid**")
                        out.append(_md_escape_cell(si.example_answer_mid))
                        out.append("")
                    if si.example_answer_bad is not None:
                        out.append("**Example Answer – Bad**")
                        out.append(_md_escape_cell(si.example_answer_bad))
                        out.append("")
                    if si.grading_rubrics is not None:
                        out.append("**Grading Rubrics**")
                        out.append(_md_escape_cell(si.grading_rubrics))
                        out.append("")
            out.append("")

    # Data quality
    out.append("## Data Quality Checks")
    out.append("")
    out.append(_md_table_kv("Quality Summary", stats.data_quality))

    return "\n".join(out).strip() + "\n"


__all__ = [
    "JudgeStats",
    "GroupCounts",
    "SampleItem",
    "ReportStats",
    "compute_report_stats",
    "render_report_md",
]
