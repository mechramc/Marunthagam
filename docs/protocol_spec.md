# Marunthagam Open Protocol v1.0

> An open, anonymized health interaction record format for community health AI systems.

---

## 1. Purpose

Rural community health systems produce valuable epidemiological signals — patterns of fever onset, respiratory illness clusters, maternal complication rates — that are invisible at the district level because they exist only in paper records, unstructured notes, or the memory of ASHA workers. Aggregating these signals can enable disease outbreak early-warning, resource pre-positioning, and evidence-based health policy. But doing so responsibly requires a format that is structured enough to be machine-readable, privacy-preserving enough to be ethically deployable, and simple enough to be implemented on a low-end Android device.

The Marunthagam Open Protocol v1.0 defines that format. It is not Marunthagam-specific. Any community health tool — in any language, any country — can adopt this schema to produce compatible interaction records. The protocol is designed to answer the question: "What is the minimum information needed to generate useful population-level health intelligence without storing anything that could identify or harm an individual patient?"

The protocol is versioned with semantic versioning. The `protocol_version` field in every record identifies the schema version that produced it, enabling receivers to handle records from multiple protocol generations simultaneously.

---

## 2. Schema

The canonical Python representation is `InteractionLogEntry` in `inference/protocol_engine/logger.py`. The JSON schema file is at `protocol/schemas/interaction_record_v1.json`.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string (UUID4) | Yes | Generated record identifier. Not linked to any patient identifier. |
| `timestamp` | string (ISO 8601 UTC) | Yes | Interaction timestamp in UTC. Example: `"2026-04-11T09:32:14.000Z"`. |
| `protocol_version` | string (semver) | Yes | Protocol version that generated this record. Current: `"1.0.0"`. |
| `locale` | string (BCP-47) | Yes | Language locale of the interaction. Example: `"ta-IN"`. |
| `device_tier` | string (enum) | Yes | Deployment tier: `"field"` (ASHA worker phone), `"clinic"` (PHC workstation), or `"district"` (dashboard). |
| `model_id` | string | Yes | Full model identifier. Example: `"gemma-4-E4B-it-marunthagam-fused-Q4_K_M"`. |
| `modalities_used` | array of strings | Yes | Modalities active during this interaction. Values: `"audio"`, `"image"`, `"text"`. |
| `triage_level` | string (enum) | Yes | Triage outcome: `"GREEN"`, `"YELLOW"`, or `"RED"`. |
| `confidence` | float (0.0–1.0) | Yes | Model's calibrated confidence in the triage level. |
| `escalation_flag` | boolean | Yes | `true` if the result was escalated due to low confidence or protocol override. |
| `protocol_overrides` | array of objects | Yes | List of protocol overrides applied by the deterministic engine. Empty array if none. Each override object contains `rule_id`, `original_level`, `overridden_to`, and `reason`. |
| `geo_hash` | string (6 chars) or null | No | 6-character geohash at ~1km precision. Optional. Null if the ASHA worker chose not to provide location. |
| `sync_status` | string (enum) | Yes | Internal sync state: `"pending"` (not yet transmitted) or `"synced"` (included in an aggregated transmission to Tier 3). |

### protocol_overrides object structure

Each entry in the `protocol_overrides` array records a single deterministic override applied by the protocol engine:

| Sub-field | Type | Description |
|-----------|------|-------------|
| `rule_id` | string | Protocol rule identifier, e.g. `"WHO-IMNCI-ARI-03"` or `"CONFIDENCE-FLOOR"`. |
| `original_level` | string | The triage level before this override. |
| `overridden_to` | string | The triage level after this override. |
| `reason` | string | Human-readable reason for the override. |

---

## 3. Privacy Guarantees

The Open Protocol is designed so that a valid record cannot contain patient-identifying information, not merely because the specification prohibits it, but because the schema has no fields that could hold it.

**What is never stored:**

- Patient name, age, address, phone number, or any demographic identifier
- Audio recordings — voice input is transcribed on-device (by Whisper-small-Tamil when used) and the audio file is discarded immediately after transcription. The transcript itself is used only for inference and is not persisted.
- Photographs or clinical images — images are processed by the multimodal model and discarded. Only the model's text interpretation (`image_findings`) is retained in memory during inference; it is not written to the interaction log.
- The symptom text, reasoning chain, or next-steps text from the triage output — these may contain contextual health details that, combined with geolocation, could narrow identification. None of these fields appear in `InteractionLogEntry`.

**Geohash precision:** The 6-character geohash format encodes location at approximately 1.2km × 0.6km cell resolution. This is sufficient for village-level disease burden mapping (a cluster of RED triage outcomes in adjacent geohash cells is a meaningful epidemiological signal) while being too coarse to identify the location of an individual household. The geohash field is optional and can be omitted entirely for communities where even 1km resolution creates identification risk.

**Local storage encryption:** The SQLite database holding interaction log records is encrypted with AES-256 on-device. The encryption key is stored in Android Keystore and never leaves the device. A device that is lost or stolen does not expose interaction records.

**Sync transmission:** The `sync_status` field tracks whether a record has been included in a Tier 3 transmission. The transmission itself does not send individual records. It sends aggregated derivatives: case counts by geohash and time bucket, escalation rates by triage level, protocol override frequencies by rule ID. An individual `InteractionLogEntry` is never transmitted over the network.

---

## 4. Aggregation Rules

The Tier 1→3 sync process aggregates interaction records before transmission. The aggregation contract is:

1. **Minimum cell size:** Aggregations over geohash cells with fewer than 5 interactions in a time bucket are suppressed. This prevents inferring health conditions from single-interaction cells that could identify a specific household.

2. **Time bucketing:** Records are aggregated by UTC day. Finer time granularity is not transmitted. The combination of day + 6-char geohash is the finest resolution that appears in the Tier 3 dataset.

3. **Aggregate fields transmitted:**
   - `date` (UTC date string)
   - `geo_hash` (6-char)
   - `case_count` (integer)
   - `red_count` (integer — RED triage outcomes)
   - `escalation_count` (integer — records with `escalation_flag = true`)
   - `override_rule_ids` (array of strings — distinct rule IDs that triggered overrides in this cell/day)
   - `model_ids` (array of strings — distinct model IDs active in this cell/day)

4. **What is not aggregated:** No symptom text, reasoning chains, or modality-specific data is included in aggregated transmissions.

5. **Retention:** Raw interaction log records on Tier 1 devices are marked `sync_status = "synced"` after their aggregated contribution has been transmitted. A configurable retention window (default: 90 days) governs how long synced records remain on-device before deletion.

---

## 5. Versioning

The protocol uses semantic versioning (`MAJOR.MINOR.PATCH`):

- **MAJOR** increment: breaking schema change — a field is removed, renamed, or its type changes incompatibly. Receivers MUST check `protocol_version` before parsing.
- **MINOR** increment: additive change — a new optional field is added. Receivers that ignore unknown fields will continue to function correctly.
- **PATCH** increment: clarification or documentation change with no schema modification.

The current version is `1.0.0`, defined as the constant `PROTOCOL_VERSION = "1.0.0"` in `inference/protocol_engine/logger.py`.

Every `InteractionLogEntry` includes the `protocol_version` field so that a Tier 3 receiver processing a batch of records from multiple device generations can apply the correct parsing logic per record. Version-aware parsing is the receiver's responsibility.

---

## 6. Example Record

A complete interaction log entry as stored in SQLite and as would appear in a JSON export:

```json
{
  "id": "a3f7c2e1-84b0-4d9a-bc3e-12f456789abc",
  "timestamp": "2026-04-11T09:32:14.000Z",
  "protocol_version": "1.0.0",
  "locale": "ta-IN",
  "device_tier": "field",
  "model_id": "gemma-4-E4B-it-marunthagam-fused-Q4_K_M",
  "modalities_used": ["audio", "text"],
  "triage_level": "RED",
  "confidence": 0.91,
  "escalation_flag": false,
  "protocol_overrides": [],
  "geo_hash": "mj7fq2",
  "sync_status": "pending"
}
```

An example with a protocol override and low-confidence escalation:

```json
{
  "id": "b8d1a409-c7f2-4e11-adf3-87e90123cdef",
  "timestamp": "2026-04-11T11:15:03.000Z",
  "protocol_version": "1.0.0",
  "locale": "ta-IN",
  "device_tier": "field",
  "model_id": "gemma-4-E4B-it-marunthagam-fused-Q4_K_M",
  "modalities_used": ["text"],
  "triage_level": "YELLOW",
  "confidence": 0.65,
  "escalation_flag": true,
  "protocol_overrides": [
    {
      "rule_id": "CONFIDENCE-FLOOR",
      "original_level": "GREEN",
      "overridden_to": "YELLOW",
      "reason": "Confidence 0.65 < 0.70 — escalate per safety protocol"
    }
  ],
  "geo_hash": "mj7fq3",
  "sync_status": "pending"
}
```

In the second example, the LLM returned GREEN with confidence 0.65. The protocol engine's confidence floor rule upgraded the outcome to YELLOW and set `escalation_flag = true`. The original model output is preserved in `protocol_overrides` for audit purposes, but the `triage_level` field reflects the final, protocol-grounded decision that was shown to the ASHA worker.
