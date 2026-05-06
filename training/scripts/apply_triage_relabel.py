"""
Apply triage GREEN→YELLOW relabels from eval/analysis/2026-05-07/_triage_relabels.json
to the actual JSONL training data, with backups + diff log.

Critical invariants enforced:
1. Backup the original split file BEFORE writing — saved alongside the live
   file as <split>_v1_pre_relabel.jsonl. Refuses to overwrite an existing
   backup (so the script is safe to re-run).
2. Each relabel touches ONLY the `tool` message's `level` field. The
   suspected_conditions, reasoning_chain, next_steps_tamil, and any other
   fields inside the tool payload are preserved verbatim. The user
   prompt, function-call args, and final assistant turn are untouched.
3. case_id maps to 1-indexed line number in the source JSONL (matching
   how the relabel CSV was generated).
4. Verifies the case's old level matches the relabel record's `old`
   field before applying — if a record claims `old=GREEN` but the file
   says YELLOW, the script aborts with a diff. No silent corrections.
5. Logs every change to eval/analysis/2026-05-07/triage_relabel_diff.json
   with case_id, line number, old/new level, and the first 200 chars of
   the user prompt for visual auditing.

Run:
    python training/scripts/apply_triage_relabel.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
RELABEL_FILE = REPO / "eval" / "analysis" / "2026-05-07" / "_triage_relabels.json"
DIFF_OUT = REPO / "eval" / "analysis" / "2026-05-07" / "triage_relabel_diff.json"
DATA_DIR = REPO / "training" / "data" / "formatted" / "triage"


def parse_case_id(case_id: str) -> tuple[str, int]:
    """Return (split, line_number) for a case_id like 'triage_train_211'."""
    parts = case_id.split("_")
    if len(parts) != 3 or parts[0] != "triage":
        raise ValueError(f"Bad case_id {case_id!r}")
    split = parts[1]
    line_num = int(parts[2])
    return split, line_num


def find_tool_level(record: dict) -> Optional[str]:
    """Return the gold level from the tool message, or None if missing."""
    for m in record.get("messages", []):
        if m.get("role") == "tool":
            content = m.get("content")
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    return None
            elif isinstance(content, dict):
                payload = content
            else:
                return None
            level = str(payload.get("level", "")).upper()
            return level if level in ("GREEN", "YELLOW", "RED") else None
    return None


def set_tool_level(record: dict, new_level: str) -> bool:
    """Mutate the tool message's level field in place. Returns True on success."""
    for m in record.get("messages", []):
        if m.get("role") == "tool":
            content = m.get("content")
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    return False
                payload["level"] = new_level
                m["content"] = json.dumps(payload, ensure_ascii=False)
                return True
            elif isinstance(content, dict):
                content["level"] = new_level
                return True
    return False


def first_user_text(record: dict, n: int = 200) -> str:
    for m in record.get("messages", []):
        if m.get("role") == "user":
            txt = m.get("content") or ""
            return txt[:n] + (" …" if len(txt) > n else "")
    return ""


def label_distribution(path: Path) -> Counter:
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            level = find_tool_level(rec)
            if level:
                counts[level] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Verify and report; do not write")
    args = parser.parse_args()

    relabel_doc = json.loads(RELABEL_FILE.read_text(encoding="utf-8"))
    changes = relabel_doc["changes"]
    print(f"Loaded {len(changes)} relabel records (date={relabel_doc.get('date')}).")

    # Group by split
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for ch in changes:
        split, line_num = parse_case_id(ch["case_id"])
        if split not in by_split:
            raise ValueError(f"Unknown split in case_id: {ch['case_id']}")
        by_split[split].append({**ch, "_line_num": line_num})

    diff_log = {"date": relabel_doc.get("date"), "splits": {}, "summary": {}}

    pre_dist: dict[str, dict] = {}
    post_dist: dict[str, dict] = {}

    for split, records in by_split.items():
        live_path = DATA_DIR / f"{split}.jsonl"
        backup_path = DATA_DIR / f"{split}_v1_pre_relabel.jsonl"

        # Pre-distribution from current file
        pre_counts = label_distribution(live_path)
        pre_dist[split] = dict(pre_counts)
        print(f"\n[{split}] before: {dict(pre_counts)}  total={sum(pre_counts.values())}")

        if not records:
            post_dist[split] = dict(pre_counts)
            diff_log["splits"][split] = []
            print(f"[{split}] no relabels for this split — skipping write.")
            continue

        # Backup
        if not args.dry_run:
            if backup_path.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing backup {backup_path}. "
                    "If you want to re-run, restore from backup first."
                )
            shutil.copy2(live_path, backup_path)
            print(f"[{split}] backed up live -> {backup_path.name}")

        # Read all lines (preserve order)
        with open(live_path, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh]

        diff_entries: list[dict] = []
        for rec in records:
            line_num = rec["_line_num"]
            idx = line_num - 1
            if idx < 0 or idx >= len(lines):
                raise IndexError(
                    f"line {line_num} out of range for {live_path} ({len(lines)} lines)"
                )
            try:
                obj = json.loads(lines[idx])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Could not parse {live_path}:{line_num}: {exc}"
                ) from exc
            current = find_tool_level(obj)
            if current != rec["old"]:
                raise RuntimeError(
                    f"Mismatch at {live_path}:{line_num} (case_id={rec['case_id']}): "
                    f"file says level={current!r}, relabel record claims old={rec['old']!r}. "
                    f"Refusing to apply silently. Verify case_id alignment."
                )
            if not set_tool_level(obj, rec["new"]):
                raise RuntimeError(f"Failed to mutate tool level at {live_path}:{line_num}")
            new_line = json.dumps(obj, ensure_ascii=False)
            diff_entries.append({
                "case_id": rec["case_id"],
                "line_number": line_num,
                "old_level": rec["old"],
                "new_level": rec["new"],
                "user_prompt_preview": first_user_text(obj),
                "rater_notes": rec.get("notes", ""),
            })
            lines[idx] = new_line

        diff_log["splits"][split] = diff_entries

        if not args.dry_run:
            with open(live_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
                fh.write("\n")
            print(f"[{split}] wrote {len(diff_entries)} relabels to {live_path.name}")

        # Post-distribution
        post_counts = label_distribution(live_path)
        post_dist[split] = dict(post_counts)
        print(f"[{split}] after:  {dict(post_counts)}  total={sum(post_counts.values())}")

    diff_log["summary"] = {
        "pre": pre_dist,
        "post": post_dist,
        "n_relabeled": sum(len(v) for v in diff_log["splits"].values()),
    }

    if not args.dry_run:
        DIFF_OUT.write_text(
            json.dumps(diff_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nDiff log written to {DIFF_OUT}")
    else:
        print(f"\n[dry-run] would write diff log to {DIFF_OUT}")
        print(json.dumps(diff_log["summary"], indent=2))


if __name__ == "__main__":
    main()
