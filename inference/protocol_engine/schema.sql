-- Marunthagam Protocol Engine Schema v2.0 (2026-05-07)
-- PRIVACY: No patient identifiers stored in any table
--
-- v2 schema replaces "regex on full narrative" matching with structured
-- chief-complaint matching to eliminate false positives on incidental
-- narrative mentions. See inference/protocol_engine/rules/imnci_rules_v2.json
-- header block for the design rationale.
--
-- Behavioural change vs v1: condition_pattern is now matched against the
-- chief complaint (verbal_symptoms) ONLY. The new required_co_signals and
-- negative_scoping columns operate on the full text (chief + narrative).

CREATE TABLE IF NOT EXISTS protocol_rules (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    -- Primary regex matched against the chief complaint (verbal_symptoms).
    -- v1 column name preserved for backwards compatibility; semantics changed.
    condition_pattern TEXT,
    -- JSON-encoded list of patterns; ALL must match anywhere in chief+narrative.
    -- NULL or '[]' means no co-signal requirement.
    required_co_signals TEXT,
    -- JSON-encoded list of patterns; rule is suppressed if ANY match in chief+narrative.
    -- NULL or '[]' means no negative scoping.
    negative_scoping TEXT,
    -- 'any' or pipe-separated set: 'adolescent|adult|elderly'
    age_group TEXT,
    duration_min_days INTEGER,
    -- New in v2: optional upper bound on duration_days (e.g., acute-onset rules).
    duration_max_days INTEGER,
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
