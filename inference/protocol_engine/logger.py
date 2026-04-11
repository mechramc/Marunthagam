"""
Log triage interactions to SQLite per Marunthagam Open Protocol v1.0.

PRIVACY GUARANTEE:
- No patient names, identifiers, or demographic data stored
- Audio and images processed ephemerally on-device; only structured result logged
- Geohash at ~1km resolution only (prevents individual tracking)
- All sync transmits aggregated signals, not individual records

Conforms to: protocol/schemas/interaction_record_v1.json
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_VERSION = "1.0.0"


@dataclass
class InteractionLogEntry:
    """Data to log for a single triage interaction. No PII fields."""
    locale: str                         # BCP-47, e.g. "ta-IN"
    device_tier: str                    # "field" | "clinic" | "district"
    model_id: str                       # e.g. "gemma-4-E4B-it-marunthagam-fused-Q4_K_M"
    modalities_used: list[str]          # e.g. ["audio", "image", "text"]
    triage_level: str                   # "GREEN" | "YELLOW" | "RED"
    confidence: float                   # 0.0–1.0
    escalation_flag: bool
    protocol_overrides: list[dict]      # overrides applied by protocol engine
    geo_hash: str | None = None         # 6-char geohash ~1km (optional)
    protocol_version: str = PROTOCOL_VERSION


class InteractionLogger:
    """
    Logs triage interactions to SQLite.

    In production: use pycryptodome for AES-256 encryption.
    In development: uses standard sqlite3 (encryption_key parameter reserved for future use).
    """

    def __init__(self, db_path: str, encryption_key: bytes | None = None) -> None:
        self.db_path = db_path
        # encryption_key reserved for pycryptodome AES-256 integration
        # TODO: implement with sqlcipher or manual encryption layer
        self._setup_db()

    def _setup_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).parent / "schema.sql"
        conn = self._connect()
        try:
            with open(schema_path) as f:
                conn.executescript(f.read())
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def log(self, entry: InteractionLogEntry) -> str:
        """
        Log a triage interaction.
        Returns the generated record_id (UUID4).
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO interaction_log
                   (id, timestamp, locale, device_tier, model_id, modalities_used,
                    triage_level, confidence, escalation_flag, protocol_overrides,
                    geo_hash, sync_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    record_id,
                    timestamp,
                    entry.locale,
                    entry.device_tier,
                    entry.model_id,
                    json.dumps(entry.modalities_used),
                    entry.triage_level,
                    entry.confidence,
                    int(entry.escalation_flag),
                    json.dumps(entry.protocol_overrides),
                    entry.geo_hash,
                    "pending",
                ],
            )
            conn.commit()
        finally:
            conn.close()

        return record_id

    def get_pending_sync(self, limit: int = 100) -> list[dict]:
        """
        Get records with sync_status='pending'.
        Used by Tier 3 sync process (transmits AGGREGATED signals only).
        """
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM interaction_log WHERE sync_status = 'pending' LIMIT ?",
                [limit],
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def mark_synced(self, record_ids: list[str]) -> None:
        """Mark records as synced after aggregate transmission to Tier 3."""
        if not record_ids:
            return
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(record_ids))
            conn.execute(
                f"UPDATE interaction_log SET sync_status = 'synced' WHERE id IN ({placeholders})",
                record_ids,
            )
            conn.commit()
        finally:
            conn.close()
