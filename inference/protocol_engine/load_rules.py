"""
Load WHO IMNCI / Tamil Nadu / Marunthagam adult-emergency protocol rules
into the SQLite database.

v2 (2026-05-07) reads the v2 schema with chief_complaint_pattern (stored as
the legacy condition_pattern column for back-compat), required_co_signals,
negative_scoping, and duration_max_days. Backwards-compatible: rules without
the new fields are loaded with empty co_signals / negative_scoping and NULL
max_days.

Usage:
    python load_rules.py --db data/protocol.db --rules rules/imnci_rules_v2.json
"""
import sqlite3
import json
import argparse
from pathlib import Path


def init_db(conn: sqlite3.Connection, schema_path: Path) -> None:
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.commit()


def _normalize(rule: dict) -> dict:
    """Translate v2 rule field names to DB column names + defaults for legacy rules."""
    # v2 JSON uses chief_complaint_pattern; DB column is condition_pattern (legacy name).
    chief = rule.get("chief_complaint_pattern", rule.get("condition_pattern"))
    co_signals = rule.get("required_co_signals", [])
    neg_scope = rule.get("negative_scoping", [])
    return {
        "id": rule["id"],
        "source": rule["source"],
        "condition_pattern": chief,
        "required_co_signals": json.dumps(co_signals, ensure_ascii=False),
        "negative_scoping": json.dumps(neg_scope, ensure_ascii=False),
        "age_group": rule.get("age_group"),
        "duration_min_days": rule.get("duration_min_days"),
        "duration_max_days": rule.get("duration_max_days"),
        "minimum_triage_level": rule["minimum_triage_level"],
        "override_reason": rule["override_reason"],
    }


def load_rules(conn: sqlite3.Connection, rules_path: Path) -> int:
    with open(rules_path, encoding="utf-8") as f:
        rules = json.load(f)

    # Skip metadata records (entries with `_id == "_metadata"` etc.)
    real_rules = [r for r in rules if not r.get("_id", "").startswith("_") and "id" in r]

    cursor = conn.cursor()
    inserted = 0
    for rule in real_rules:
        n = _normalize(rule)
        cursor.execute(
            """INSERT OR IGNORE INTO protocol_rules
               (id, source, condition_pattern, required_co_signals,
                negative_scoping, age_group, duration_min_days,
                duration_max_days, minimum_triage_level,
                override_reason, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            [
                n["id"], n["source"], n["condition_pattern"],
                n["required_co_signals"], n["negative_scoping"],
                n["age_group"], n["duration_min_days"], n["duration_max_days"],
                n["minimum_triage_level"], n["override_reason"],
            ]
        )
        if cursor.rowcount > 0:
            inserted += 1
    conn.commit()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed protocol grounding engine database")
    parser.add_argument("--db", default="data/protocol.db", help="SQLite database path")
    parser.add_argument("--rules", default="rules/imnci_rules_v2.json",
                        help="Rules JSON path (v2 schema; v1 also accepted)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing DB before loading (forces full reload)")
    args = parser.parse_args()

    db_path = Path(args.db)
    rules_path = Path(args.rules)
    schema_path = Path(__file__).parent / "schema.sql"

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    if args.reset and db_path.exists():
        db_path.unlink()
        print(f"[reset] removed {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    try:
        init_db(conn, schema_path)
        inserted = load_rules(conn, rules_path)
        total = conn.execute(
            "SELECT COUNT(*) FROM protocol_rules WHERE active = 1"
        ).fetchone()[0]
        print(f"Inserted {inserted} new rules. Total active rules: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
