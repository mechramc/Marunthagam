"""Tests for the Marunthagam interaction logger."""
import json
import sqlite3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from logger import InteractionLogger, InteractionLogEntry, PROTOCOL_VERSION


def make_entry(**overrides) -> InteractionLogEntry:
    base = dict(
        locale="ta-IN",
        device_tier="field",
        model_id="gemma-4-E4B-it-test",
        modalities_used=["text"],
        triage_level="GREEN",
        confidence=0.88,
        escalation_flag=False,
        protocol_overrides=[],
    )
    base.update(overrides)
    return InteractionLogEntry(**base)


@pytest.fixture
def logger(tmp_path):
    db = str(tmp_path / "test.db")
    return InteractionLogger(db)


class TestLogCreation:
    def test_log_returns_uuid_string(self, logger):
        record_id = logger.log(make_entry())
        assert isinstance(record_id, str)
        assert len(record_id) == 36  # UUID4 format: 8-4-4-4-12

    def test_log_record_retrievable(self, logger):
        record_id = logger.log(make_entry(triage_level="YELLOW"))
        conn = sqlite3.connect(logger.db_path)
        row = conn.execute(
            "SELECT triage_level FROM interaction_log WHERE id = ?", [record_id]
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "YELLOW"

    def test_multiple_logs_produce_distinct_uuids(self, logger):
        ids = [logger.log(make_entry()) for _ in range(5)]
        assert len(set(ids)) == 5, "All 5 UUIDs must be distinct"


class TestPiiGuarantee:
    def test_no_patient_pii_columns_in_schema(self, logger):
        """The interaction_log table must never have PII columns."""
        conn = sqlite3.connect(logger.db_path)
        columns = [row[1] for row in conn.execute(
            "PRAGMA table_info(interaction_log)"
        ).fetchall()]
        conn.close()
        forbidden = {
            "patient_name", "patient_id", "patient_phone",
            "name", "dob", "phone", "address", "nric",
            "age", "gender", "location",
        }
        found = forbidden.intersection(set(columns))
        assert not found, f"PII columns found in schema: {found}"


class TestDataStorage:
    def test_modalities_stored_as_json_array(self, logger):
        logger.log(make_entry(modalities_used=["audio", "image"]))
        conn = sqlite3.connect(logger.db_path)
        raw = conn.execute(
            "SELECT modalities_used FROM interaction_log"
        ).fetchone()[0]
        conn.close()
        parsed = json.loads(raw)
        assert parsed == ["audio", "image"]

    def test_geo_hash_nullable(self, logger):
        record_id = logger.log(make_entry(geo_hash=None))
        conn = sqlite3.connect(logger.db_path)
        geo = conn.execute(
            "SELECT geo_hash FROM interaction_log WHERE id = ?", [record_id]
        ).fetchone()[0]
        conn.close()
        assert geo is None

    def test_geo_hash_stored_when_provided(self, logger):
        record_id = logger.log(make_entry(geo_hash="t8h5rt"))
        conn = sqlite3.connect(logger.db_path)
        geo = conn.execute(
            "SELECT geo_hash FROM interaction_log WHERE id = ?", [record_id]
        ).fetchone()[0]
        conn.close()
        assert geo == "t8h5rt"


class TestSyncStatus:
    def test_default_sync_status_is_pending(self, logger):
        record_id = logger.log(make_entry())
        conn = sqlite3.connect(logger.db_path)
        status = conn.execute(
            "SELECT sync_status FROM interaction_log WHERE id = ?", [record_id]
        ).fetchone()[0]
        conn.close()
        assert status == "pending"

    def test_mark_synced_updates_status(self, logger):
        record_id = logger.log(make_entry())
        logger.mark_synced([record_id])
        conn = sqlite3.connect(logger.db_path)
        status = conn.execute(
            "SELECT sync_status FROM interaction_log WHERE id = ?", [record_id]
        ).fetchone()[0]
        conn.close()
        assert status == "synced"

    def test_mark_synced_only_affects_specified_ids(self, logger):
        id1 = logger.log(make_entry())
        id2 = logger.log(make_entry())
        logger.mark_synced([id1])
        conn = sqlite3.connect(logger.db_path)
        status2 = conn.execute(
            "SELECT sync_status FROM interaction_log WHERE id = ?", [id2]
        ).fetchone()[0]
        conn.close()
        assert status2 == "pending", "id2 should still be pending"

    def test_get_pending_sync_returns_pending_records(self, logger):
        id1 = logger.log(make_entry(triage_level="GREEN"))
        id2 = logger.log(make_entry(triage_level="RED"))
        logger.mark_synced([id1])
        pending = logger.get_pending_sync()
        pending_ids = {r["id"] for r in pending}
        assert id2 in pending_ids, "Unsynced record must appear in get_pending_sync"
        assert id1 not in pending_ids, "Synced record must not appear in get_pending_sync"

    def test_get_pending_sync_respects_limit(self, logger):
        for _ in range(5):
            logger.log(make_entry())
        pending = logger.get_pending_sync(limit=3)
        assert len(pending) <= 3, "get_pending_sync must respect the limit parameter"
