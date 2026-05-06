"""
Per-rule firing audit on the held-out test split.

Critical observability gap surfaced in Sprint 2 follow-up (2026-05-07):
the existing `engine_overrides` log captures only ESCALATING matches —
rules whose minimum_triage_level is higher than the current level when
checked. A rule that matches but doesn't escalate (e.g. ADULT-CARDIAC-001
fires on a case where the model already said RED) does NOT appear in
overrides. The user's per-rule audit requires every match.

This script is read-only and does not touch engine.py. It re-runs rule
matching against each held-out test case, applies the v2 _matches_rule
logic directly, and reports for each rule:

    total_matches        — rule fired (all filters passed)
    by_gold_class        — { GREEN: n, YELLOW: n, RED: n }
    true_positive        — rule with minimum_level == gold_class
    false_positive       — rule with minimum_level > gold_class (over-trigger)
    chief_only_misses    — chief_complaint_pattern matched on verbal_symptoms
                            BUT rule still didn't fire (debugging signal —
                            why did the rule not match overall?)
                            Each entry includes which filter rejected it.

Output:
    eval/analysis/2026-05-07/_rule_firing_audit.json   (raw per-rule + per-case)
    Returns a Python dict for direct use by the analysis memo.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "inference" / "protocol_engine" / "data" / "protocol.db"
OUT = REPO / "eval" / "analysis" / "2026-05-07" / "_rule_firing_audit.json"


def load_rules() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM protocol_rules WHERE active=1").fetchall()
    conn.close()
    rules = []
    for row in rows:
        rules.append({
            "id": row["id"],
            "source": row["source"],
            "condition_pattern": row["condition_pattern"],
            "required_co_signals": _decode_list(row["required_co_signals"]),
            "negative_scoping": _decode_list(row["negative_scoping"]),
            "age_group": row["age_group"],
            "duration_min_days": row["duration_min_days"],
            "duration_max_days": row["duration_max_days"],
            "minimum_triage_level": row["minimum_triage_level"],
            "override_reason": row["override_reason"],
        })
    return rules


def _decode_list(raw):
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


def load_test_cases() -> list[dict]:
    """Load held-out test split with chief_complaint + narrative + age + duration."""
    cases = []
    for spec in ("triage", "derm", "maternal"):
        path = REPO / "training" / "data" / "formatted" / spec / "test.jsonl"
        for line_num, line in enumerate(open(path, encoding="utf-8"), start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            tamil_q = ""
            args = {}
            tool = {}
            for m in messages:
                r = m.get("role")
                if r == "user" and not tamil_q:
                    tamil_q = m.get("content") or ""
                elif r == "assistant" and m.get("tool_calls"):
                    raw = m["tool_calls"][0].get("function", {}).get("arguments")
                    if isinstance(raw, str):
                        try: args = json.loads(raw)
                        except: pass
                    elif isinstance(raw, dict):
                        args = raw
                elif r == "tool":
                    raw = m.get("content")
                    if isinstance(raw, str):
                        try: tool = json.loads(raw)
                        except: pass
                    elif isinstance(raw, dict):
                        tool = raw
            level = str(tool.get("level", "")).upper()
            if level not in ("GREEN", "YELLOW", "RED"):
                continue
            cases.append({
                "case_id": f"{spec}_test_{line_num:03d}",
                "specialist": spec,
                "verbal_symptoms": (args.get("verbal_symptoms") or "").strip(),
                "tamil_question": (tamil_q or "").strip(),
                "age_group": args.get("patient_age_group", "adult"),
                "duration_days": int(args.get("duration_days", 1) or 1),
                "gold": level,
            })
    return cases


def _age_matches(rule_age: str | None, patient_age: str) -> bool:
    if not rule_age or rule_age.lower() == "any":
        return True
    return patient_age.lower() in {a.strip().lower() for a in rule_age.split("|") if a.strip()}


def evaluate_rule(rule: dict, case: dict) -> dict:
    """
    Run rule against a case and return a structured filter-by-filter trace.

    fired = True means ALL filters passed. If False, `failed_filters` lists
    which one(s) rejected the case.
    """
    chief = case["verbal_symptoms"]
    narrative = case["tamil_question"]
    full_text = f"{chief}\n{narrative}" if narrative else chief
    age = case["age_group"]
    duration = case["duration_days"]

    failed = []
    chief_match = True
    if rule["condition_pattern"]:
        try:
            chief_match = bool(re.search(rule["condition_pattern"], chief, re.IGNORECASE))
        except re.error:
            chief_match = False
    if not chief_match:
        failed.append("chief_complaint_pattern")

    co_signals_pass = True
    co_failures: list[str] = []
    for pat in rule["required_co_signals"]:
        try:
            ok = bool(re.search(pat, full_text, re.IGNORECASE))
        except re.error:
            ok = False
        if not ok:
            co_failures.append(pat[:60] + ("…" if len(pat) > 60 else ""))
            co_signals_pass = False
    if not co_signals_pass:
        failed.append(f"required_co_signals ({len(co_failures)} unmatched)")

    neg_match = False
    neg_hits: list[str] = []
    for pat in rule["negative_scoping"]:
        try:
            if re.search(pat, full_text, re.IGNORECASE):
                neg_hits.append(pat[:60])
                neg_match = True
        except re.error:
            continue
    if neg_match:
        failed.append(f"negative_scoping (matched: {neg_hits})")

    age_ok = _age_matches(rule["age_group"], age)
    if not age_ok:
        failed.append(f"age_group (rule={rule['age_group']!r} patient={age!r})")

    duration_ok = True
    if rule["duration_min_days"] is not None and duration < rule["duration_min_days"]:
        duration_ok = False
        failed.append(f"duration_min_days (rule={rule['duration_min_days']} case={duration})")
    if rule["duration_max_days"] is not None and duration > rule["duration_max_days"]:
        duration_ok = False
        failed.append(f"duration_max_days (rule={rule['duration_max_days']} case={duration})")

    return {
        "rule_id": rule["id"],
        "min_level": rule["minimum_triage_level"],
        "fired": len(failed) == 0,
        "chief_pattern_matched": chief_match,
        "failed_filters": failed,
        "co_signal_failures": co_failures,
        "negative_scoping_hits": neg_hits,
    }


def main() -> None:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    rules = load_rules()
    cases = load_test_cases()
    print(f"Loaded {len(rules)} rules and {len(cases)} held-out test cases.\n")

    # Per-rule + per-case audit
    per_rule = {r["id"]: {
        "id": r["id"],
        "source": r["source"],
        "min_level": r["minimum_triage_level"],
        "fires": [],                 # case_ids where the rule fired
        "fires_by_gold": Counter(),  # {GREEN: n, YELLOW: n, RED: n}
        "chief_only_misses": [],     # rule didn't fire BUT chief regex did match
    } for r in rules}

    per_case = []
    for case in cases:
        per_case_rules = []
        for rule in rules:
            ev = evaluate_rule(rule, case)
            per_case_rules.append(ev)
            agg = per_rule[rule["id"]]
            if ev["fired"]:
                agg["fires"].append(case["case_id"])
                agg["fires_by_gold"][case["gold"]] += 1
            elif ev["chief_pattern_matched"]:
                agg["chief_only_misses"].append({
                    "case_id": case["case_id"],
                    "gold": case["gold"],
                    "failed_filters": ev["failed_filters"],
                })
        per_case.append({
            "case_id": case["case_id"],
            "gold": case["gold"],
            "specialist": case["specialist"],
            "age_group": case["age_group"],
            "duration_days": case["duration_days"],
            "verbal_symptoms": case["verbal_symptoms"],
            "tamil_question": case["tamil_question"][:300] + ("…" if len(case["tamil_question"]) > 300 else ""),
            "rule_evaluations": per_case_rules,
        })

    # Print summary
    print("=== Per-rule firing summary (held-out n=" + str(len(cases)) + ") ===\n")
    print(f"{'rule_id':<30} {'min':<6} {'total':<6} G  Y  R   chief_only_miss  TP  FP")
    new_rule_ids = {
        "ADULT-CARDIAC-001", "ADULT-ANAPHYLAXIS-001", "ADULT-HEAD-TRAUMA-001",
        "ADULT-RESPIRATORY-001", "ANIMAL-BITE-RESPIRATORY-001",
        "NEW-ONSET-JAUNDICE-001",
    }
    for rid in sorted(per_rule.keys()):
        agg = per_rule[rid]
        total = len(agg["fires"])
        g = agg["fires_by_gold"].get("GREEN", 0)
        y = agg["fires_by_gold"].get("YELLOW", 0)
        r_ = agg["fires_by_gold"].get("RED", 0)
        miss = len(agg["chief_only_misses"])
        # TP = case gold matches the rule's escalation target (or is RED for RED rules)
        tp = r_ if agg["min_level"] == "RED" else (y + r_) if agg["min_level"] == "YELLOW" else 0
        fp = total - tp
        marker = " ★" if rid in new_rule_ids else ""
        print(f"{rid:<30}{marker} {agg['min_level']:<6} {total:<6} {g:<2} {y:<2} {r_:<2}   {miss:<14}  {tp:<3} {fp:<3}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({
            "n_cases": len(cases),
            "n_rules": len(rules),
            "per_rule": {
                rid: {
                    **{k: v for k, v in agg.items() if k != "fires_by_gold"},
                    "fires_by_gold": dict(agg["fires_by_gold"]),
                }
                for rid, agg in per_rule.items()
            },
            "per_case": per_case,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull audit saved to {OUT}")


if __name__ == "__main__":
    main()
