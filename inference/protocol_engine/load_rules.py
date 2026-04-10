"""
Load WHO IMNCI and Tamil Nadu protocol rules into the SQLite database.
Run this once to seed the protocol grounding engine database.

Usage:
    python load_rules.py --db data/protocol.db --rules rules/imnci_rules.json
"""
import sqlite3
import json
import argparse
from pathlib import Path


def init_db(conn: sqlite3.Connection, schema_path: Path) -> None:
    """Run schema.sql to create tables if they don't exist."""
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.commit()


def load_rules(conn: sqlite3.Connection, rules_path: Path) -> int:
    """
    Load rules from JSON file into protocol_rules table.
    Uses INSERT OR IGNORE to skip duplicates (idempotent).
    Returns count of newly inserted rules.
    """
    with open(rules_path, encoding="utf-8") as f:
        rules = json.load(f)

    cursor = conn.cursor()
    inserted = 0
    for rule in rules:
        cursor.execute(
            """INSERT OR IGNORE INTO protocol_rules
               (id, source, condition_pattern, age_group, duration_min_days,
                minimum_triage_level, override_reason, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
            [
                rule["id"],
                rule["source"],
                rule.get("condition_pattern"),
                rule.get("age_group"),
                rule.get("duration_min_days"),
                rule["minimum_triage_level"],
                rule["override_reason"],
            ]
        )
        if cursor.rowcount > 0:
            inserted += 1
    conn.commit()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed protocol grounding engine database")
    parser.add_argument("--db", default="data/protocol.db", help="SQLite database path")
    parser.add_argument("--rules", default="rules/imnci_rules.json", help="Rules JSON path")
    args = parser.parse_args()

    db_path = Path(args.db)
    rules_path = Path(args.rules)
    schema_path = Path(__file__).parent / "schema.sql"

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    try:
        init_db(conn, schema_path)
        inserted = load_rules(conn, rules_path)
        total = conn.execute("SELECT COUNT(*) FROM protocol_rules WHERE active = 1").fetchone()[0]
        print(f"Inserted {inserted} new rules. Total active rules: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
