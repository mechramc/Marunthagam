"""
Apply user-determined derm contamination verdicts from
eval/analysis/2026-05-07/_derm_verdicts.json to the actual JSONL data.

Verdicts (one per row in derm_contamination_candidates.csv, same order):
  KEEP_IN_DERM      → no-op
  RELABEL_ONLY      → mutate the derm row's tool message level (uses new_label)
  MOVE_TO_TRIAGE    → remove from derm, append to corresponding triage split
                       (with optional level relabel via new_label)
  DROP              → remove from derm, do not append anywhere

Critical invariants enforced:
1. ALL six files (derm/{train,val,test}.jsonl + triage/{train,val,test}.jsonl)
   are backed up to <split>_v2_pre_derm_move.jsonl BEFORE any write.
   Refuses to overwrite an existing v2 backup.
2. Verdicts are paired with CSV rows by ORDER. The CSV's case_ids encode
   <split>_<line_number>; we honour that for both source (derm) and
   destination (which triage split a moved case lands in).
3. When a case is moved to triage, the NEW case_id reflects its new line
   number in triage/<split>.jsonl after append. Diff log preserves both.
4. Each level mutation touches ONLY the tool message's `level` field;
   suspected_conditions, reasoning_chain, etc. are preserved verbatim.
5. Diff log at eval/analysis/2026-05-07/derm_contamination_diff.json
   records every action with verdict, case_id, old_level, new_level,
   user prompt preview, and rater notes.
6. Script verifies the verdict count exactly matches the CSV row count
   before doing anything — refuses to apply on a length mismatch.

Run:
    python training/scripts/apply_derm_contamination.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
VERDICTS_FILE = REPO / "eval" / "analysis" / "2026-05-07" / "_derm_verdicts.json"
CSV_FILE = REPO / "eval" / "analysis" / "2026-05-07" / "derm_contamination_candidates.csv"
DIFF_OUT = REPO / "eval" / "analysis" / "2026-05-07" / "derm_contamination_diff.json"
DERM_DIR = REPO / "training" / "data" / "formatted" / "derm"
TRIAGE_DIR = REPO / "training" / "data" / "formatted" / "triage"

VALID_VERDICTS = {"KEEP_IN_DERM", "RELABEL_ONLY", "MOVE_TO_TRIAGE", "DROP"}
VALID_LEVELS = {"GREEN", "YELLOW", "RED"}


def parse_case_id(case_id: str) -> tuple[str, str, int]:
    parts = case_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad case_id {case_id!r}")
    return parts[0], parts[1], int(parts[2])


def find_tool_level(record: dict) -> Optional[str]:
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
            return level if level in VALID_LEVELS else None
    return None


def set_tool_level(record: dict, new_level: str) -> bool:
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
    if not path.exists():
        return counts
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


def backup_split(split_path: Path, suffix: str) -> Path:
    backup = split_path.with_name(split_path.stem + suffix + split_path.suffix)
    if backup.exists():
        raise RuntimeError(f"Refusing to overwrite existing backup {backup}.")
    shutil.copy2(split_path, backup)
    return backup


def load_jsonl(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def write_jsonl(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load verdicts
    doc = json.loads(VERDICTS_FILE.read_text(encoding="utf-8"))
    verdicts = doc["verdicts"]

    # Load CSV (must be in same order as verdicts)
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))

    if len(csv_rows) != len(verdicts):
        raise RuntimeError(
            f"Length mismatch: csv has {len(csv_rows)} rows, verdicts has {len(verdicts)}. "
            "Refusing to apply."
        )

    # Validate every verdict
    for i, v in enumerate(verdicts):
        if v["verdict"] not in VALID_VERDICTS:
            raise ValueError(f"Row {i}: bad verdict {v['verdict']!r}")
        if v["new_label"] is not None and v["new_label"] not in VALID_LEVELS:
            raise ValueError(f"Row {i}: bad new_label {v['new_label']!r}")

    # Pair CSV row with verdict and case_id
    paired = []
    for csv_row, v in zip(csv_rows, verdicts):
        case_id = csv_row["case_id"]
        spec, split, line_num = parse_case_id(case_id)
        if spec != "derm":
            raise ValueError(f"Unexpected case_id {case_id!r} — expected derm prefix")
        paired.append({
            "case_id": case_id,
            "split": split,
            "line_num": line_num,
            "verdict": v["verdict"],
            "new_label": v["new_label"],
            "notes": v["notes"],
            "csv_row": csv_row,
        })

    # Group by split for batched processing
    by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    for p in paired:
        by_split[p["split"]].append(p)

    # Pre-distributions
    pre_dist: dict[str, dict[str, dict]] = {"derm": {}, "triage": {}}
    for split in ("train", "val", "test"):
        pre_dist["derm"][split] = dict(label_distribution(DERM_DIR / f"{split}.jsonl"))
        pre_dist["triage"][split] = dict(label_distribution(TRIAGE_DIR / f"{split}.jsonl"))

    print("Pre-distributions:")
    for spec in ("derm", "triage"):
        for split in ("train", "val", "test"):
            d = pre_dist[spec][split]
            print(f"  {spec}/{split}: {d}  total={sum(d.values())}")
    print()

    # Backup all 6 files unless dry-run
    backup_paths: dict[Path, Path] = {}
    if not args.dry_run:
        for d in (DERM_DIR, TRIAGE_DIR):
            for split in ("train", "val", "test"):
                p = d / f"{split}.jsonl"
                bp = backup_split(p, "_v2_pre_derm_move")
                backup_paths[p] = bp
                print(f"backed up {p.name} -> {bp.name}")
        print()

    # Process each split
    diff_log: dict = {
        "date": doc.get("date"),
        "splits": {},
        "summary": {},
    }
    counts: Counter = Counter()

    for split, items in by_split.items():
        derm_path = DERM_DIR / f"{split}.jsonl"
        triage_path = TRIAGE_DIR / f"{split}.jsonl"
        derm_lines = load_jsonl(derm_path)
        triage_lines = load_jsonl(triage_path)

        # Sort items by line_num descending so deletions don't shift indices.
        # We collect actions first, then apply removals back-to-front.
        actions: list[dict] = []  # records keyed by source line_num
        for item in items:
            idx = item["line_num"] - 1
            if idx < 0 or idx >= len(derm_lines):
                raise IndexError(
                    f"line {item['line_num']} OOB for {derm_path} ({len(derm_lines)} lines)"
                )
            try:
                obj = json.loads(derm_lines[idx])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Bad JSON {derm_path}:{item['line_num']}: {exc}") from exc

            current_level = find_tool_level(obj)
            verdict = item["verdict"]

            entry = {
                "case_id": item["case_id"],
                "verdict": verdict,
                "split": split,
                "source_line_num": item["line_num"],
                "old_level": current_level,
                "new_level": current_level,
                "user_prompt_preview": first_user_text(obj),
                "rater_notes": item["notes"],
                "destination": None,
                "destination_line_num": None,
            }

            if verdict == "KEEP_IN_DERM":
                entry["destination"] = "derm (unchanged)"
            elif verdict == "DROP":
                entry["destination"] = "DROPPED"
                actions.append({"kind": "remove_from_derm", "src_idx": idx, "entry": entry})
            elif verdict == "RELABEL_ONLY":
                if item["new_label"] is None:
                    raise ValueError(f"{item['case_id']}: RELABEL_ONLY requires new_label")
                # Mutate in place
                if not set_tool_level(obj, item["new_label"]):
                    raise RuntimeError(f"Failed to set level on {item['case_id']}")
                derm_lines[idx] = json.dumps(obj, ensure_ascii=False)
                entry["new_level"] = item["new_label"]
                entry["destination"] = "derm (relabeled in place)"
            elif verdict == "MOVE_TO_TRIAGE":
                if item["new_label"] is not None:
                    if not set_tool_level(obj, item["new_label"]):
                        raise RuntimeError(f"Failed to set level on {item['case_id']}")
                    entry["new_level"] = item["new_label"]
                # The actual append (and removal from derm) is done after we've
                # collected all actions, in a deterministic order.
                actions.append({
                    "kind": "move_to_triage",
                    "src_idx": idx,
                    "obj_serialised": json.dumps(obj, ensure_ascii=False),
                    "entry": entry,
                })
            diff_log["splits"].setdefault(split, []).append(entry)
            counts[verdict] += 1

        # Apply moves: append to triage first (so we can capture the destination
        # line number), then remove from derm in reverse order.
        moves = [a for a in actions if a["kind"] == "move_to_triage"]
        drops = [a for a in actions if a["kind"] == "remove_from_derm"]

        # Append to triage in original CSV order so destination line numbers
        # are deterministic. Skip if destination already exceeds bounds (it
        # won't, we just append).
        for action in moves:
            triage_lines.append(action["obj_serialised"])
            action["entry"]["destination"] = f"triage/{split}.jsonl"
            action["entry"]["destination_line_num"] = len(triage_lines)

        # Remove from derm in reverse line order so earlier indices are stable.
        to_remove_indices = sorted(
            [a["src_idx"] for a in (moves + drops)],
            reverse=True,
        )
        for src_idx in to_remove_indices:
            del derm_lines[src_idx]

        if not args.dry_run:
            write_jsonl(derm_path, derm_lines)
            write_jsonl(triage_path, triage_lines)

    # Post-distributions
    post_dist: dict[str, dict[str, dict]] = {"derm": {}, "triage": {}}
    for split in ("train", "val", "test"):
        post_dist["derm"][split] = dict(label_distribution(DERM_DIR / f"{split}.jsonl"))
        post_dist["triage"][split] = dict(label_distribution(TRIAGE_DIR / f"{split}.jsonl"))

    diff_log["summary"] = {
        "verdict_counts": dict(counts),
        "pre": pre_dist,
        "post": post_dist,
    }

    if not args.dry_run:
        DIFF_OUT.write_text(json.dumps(diff_log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nVerdict counts:", dict(counts))
    print("\nPost-distributions:")
    for spec in ("derm", "triage"):
        for split in ("train", "val", "test"):
            d = post_dist[spec][split]
            pre = pre_dist[spec][split]
            delta = {k: d.get(k, 0) - pre.get(k, 0) for k in set(d) | set(pre)}
            print(f"  {spec}/{split}: {d}  delta={delta}  total={sum(d.values())}")
    print(f"\n{'[dry-run] would write' if args.dry_run else 'Wrote'} diff log to {DIFF_OUT}")


if __name__ == "__main__":
    main()
