-- Marunthagam Protocol Engine Schema v1.0
-- PRIVACY: No patient identifiers stored in any table

CREATE TABLE IF NOT EXISTS protocol_rules (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    condition_pattern TEXT,
    age_group TEXT,
    duration_min_days INTEGER,
    minimum_triage_level TEXT NOT NULL CHECK (minimum_triage_level IN ('GREEN', 'YELLOW', 'RED')),
    override_reason TEXT NOT NULL,
    active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS interaction_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    locale TEXT NOT NULL,
    device_tier TEXT NOT NULL,
    model_id TEXT NOT NULL,
    modalities_used TEXT NOT NULL,
    triage_level TEXT NOT NULL CHECK (triage_level IN ('GREEN', 'YELLOW', 'RED')),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    escalation_flag INTEGER NOT NULL CHECK (escalation_flag IN (0, 1)),
    protocol_overrides TEXT,
    geo_hash TEXT,
    sync_status TEXT NOT NULL DEFAULT 'pending' CHECK (sync_status IN ('pending', 'synced', 'failed')),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_interaction_log_sync ON interaction_log(sync_status);
CREATE INDEX IF NOT EXISTS idx_interaction_log_timestamp ON interaction_log(timestamp);
