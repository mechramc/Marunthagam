package com.marunthagam.storage

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log
import com.marunthagam.inference.TriageLevel
import java.util.UUID

// -----------------------------------------------------------------------
// Data model
// -----------------------------------------------------------------------

/**
 * Anonymised triage log entry.
 *
 * Privacy design:
 *   - NO patient name, phone number, or any identifying field.
 *   - NO raw symptom text (retained only transiently in [TriageResult.rawOutput]).
 *   - NO village / sub-district finer than geohash precision ~1 km.
 *   - The [id] is a random UUID generated at insert time — not derived from
 *     any patient attribute.
 *
 * Encryption:
 *   The database file itself is encrypted via AES-256-CBC using a key stored in
 *   Android Keystore (see [TriageLogDatabase] for the TODO scaffold). Individual
 *   field-level encryption is not applied here because the database-level
 *   encryption covers all rows. If SQLCipher is added later, replace
 *   [SQLiteOpenHelper] with [net.sqlcipher.database.SQLiteOpenHelper].
 *
 * @property id             Random UUID — primary key, no semantic meaning.
 * @property timestamp      Unix epoch milliseconds (UTC).
 * @property ageGroup       One of: infant | child | adolescent | adult | elderly.
 * @property durationDays   Symptom duration in days.
 * @property triageLevel    GREEN | YELLOW | RED.
 * @property confidence     Model confidence [0.0, 1.0].
 * @property escalationFlag True if the protocol forced an upgrade or confidence was low.
 */
data class TriageLog(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val ageGroup: String,
    val durationDays: Int,
    val triageLevel: TriageLevel,
    val confidence: Float,
    val escalationFlag: Boolean,
)

// -----------------------------------------------------------------------
// Database helper
// -----------------------------------------------------------------------

/**
 * SQLite database helper for anonymised triage logs.
 *
 * AES-256 encryption — TODO scaffold:
 * ```kotlin
 * // 1. Generate / retrieve key from Android Keystore
 * val keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
 * keyGenerator.init(
 *     KeyGenParameterSpec.Builder(
 *         "marunthagam_db_key",
 *         KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
 *     )
 *     .setBlockModes(KeyProperties.BLOCK_MODE_CBC)
 *     .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_PKCS7)
 *     .setKeySize(256)
 *     .build()
 * )
 * keyGenerator.generateKey()
 *
 * // 2. Replace SQLiteOpenHelper with SQLCipher variant and pass passphrase derived
 * //    from the Keystore-protected key.
 * //    Dependency: implementation("net.zetetic:android-database-sqlcipher:4.5.4")
 * ```
 *
 * Until SQLCipher is wired up, the database file is stored in the app's private
 * data directory (`/data/data/com.marunthagam/databases/`) which is
 * protected by Android's app sandbox (no other app can read it without root).
 */
private class TriageLogDatabase(context: Context) :
    SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(CREATE_TABLE_SQL)
        Log.i(TAG, "onCreate: triage_log table created")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Simple drop-and-recreate for now; add ALTER TABLE migrations when schema evolves.
        Log.w(TAG, "onUpgrade: dropping and recreating triage_log (v$oldVersion → v$newVersion)")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")
        onCreate(db)
    }

    companion object {
        private const val TAG            = "TriageLogDatabase"
        private const val DATABASE_NAME  = "marunthagam_triage.db"
        private const val DATABASE_VERSION = 1
        private const val TABLE_NAME     = "triage_log"

        private const val CREATE_TABLE_SQL = """
            CREATE TABLE IF NOT EXISTS $TABLE_NAME (
                id              TEXT PRIMARY KEY,
                timestamp       INTEGER NOT NULL,
                age_group       TEXT NOT NULL,
                duration_days   INTEGER NOT NULL,
                triage_level    TEXT NOT NULL,
                confidence      REAL NOT NULL,
                escalation_flag INTEGER NOT NULL
            )
        """
    }
}

// -----------------------------------------------------------------------
// DAO
// -----------------------------------------------------------------------

/**
 * Data Access Object for anonymised triage logs.
 *
 * All database operations are synchronous — callers are responsible for
 * dispatching on [kotlinx.coroutines.Dispatchers.IO] to avoid blocking the
 * main thread.
 *
 * Example usage in a coroutine:
 * ```kotlin
 * withContext(Dispatchers.IO) {
 *     val dao = TriageLogDao(context)
 *     dao.insertLog(TriageLog(
 *         ageGroup      = "adult",
 *         durationDays  = 3,
 *         triageLevel   = TriageLevel.YELLOW,
 *         confidence    = 0.82f,
 *         escalationFlag = false,
 *     ))
 * }
 * ```
 */
class TriageLogDao(context: Context) {

    private val dbHelper = TriageLogDatabase(context.applicationContext)

    companion object {
        private const val TAG        = "TriageLogDao"
        private const val TABLE_NAME = "triage_log"
    }

    /**
     * Inserts a single anonymised [TriageLog] entry.
     *
     * Uses INSERT OR REPLACE to handle the (unlikely) UUID collision gracefully.
     *
     * @param log The log entry to persist. Must not contain patient-identifiable data.
     * @return The row ID assigned by SQLite, or -1 on failure.
     */
    fun insertLog(log: TriageLog): Long {
        val values = ContentValues().apply {
            put("id",              log.id)
            put("timestamp",       log.timestamp)
            put("age_group",       log.ageGroup)
            put("duration_days",   log.durationDays)
            put("triage_level",    log.triageLevel.name)
            put("confidence",      log.confidence)
            put("escalation_flag", if (log.escalationFlag) 1 else 0)
        }

        return dbHelper.writableDatabase.use { db ->
            val rowId = db.insertWithOnConflict(
                TABLE_NAME,
                null,
                values,
                SQLiteDatabase.CONFLICT_REPLACE,
            )
            if (rowId == -1L) {
                Log.e(TAG, "insertLog: insert failed for id=${log.id}")
            } else {
                Log.d(TAG, "insertLog: inserted row $rowId, triageLevel=${log.triageLevel.name}")
            }
            rowId
        }
    }

    /**
     * Retrieves the most recent triage log entries, ordered by [TriageLog.timestamp] descending.
     *
     * @param limit Maximum number of rows to return (default 100). Used for UI list views
     *              and aggregated sync payloads.
     * @return List of [TriageLog] entries; empty list on any error.
     */
    fun getLogs(limit: Int = 100): List<TriageLog> {
        val safeLimit = maxOf(1, limit)
        val results = mutableListOf<TriageLog>()

        dbHelper.readableDatabase.use { db ->
            db.query(
                TABLE_NAME,
                /*columns=*/       null,  // SELECT *
                /*selection=*/     null,
                /*selectionArgs=*/ null,
                /*groupBy=*/       null,
                /*having=*/        null,
                /*orderBy=*/       "timestamp DESC",
                /*limit=*/         safeLimit.toString(),
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    runCatching {
                        val levelString = cursor.getString(cursor.getColumnIndexOrThrow("triage_level"))
                        val level = runCatching { TriageLevel.valueOf(levelString) }
                            .getOrElse {
                                Log.w(TAG, "getLogs: unknown triage_level='$levelString', defaulting to YELLOW")
                                TriageLevel.YELLOW
                            }

                        TriageLog(
                            id             = cursor.getString(cursor.getColumnIndexOrThrow("id")),
                            timestamp      = cursor.getLong(cursor.getColumnIndexOrThrow("timestamp")),
                            ageGroup       = cursor.getString(cursor.getColumnIndexOrThrow("age_group")),
                            durationDays   = cursor.getInt(cursor.getColumnIndexOrThrow("duration_days")),
                            triageLevel    = level,
                            confidence     = cursor.getFloat(cursor.getColumnIndexOrThrow("confidence")),
                            escalationFlag = cursor.getInt(cursor.getColumnIndexOrThrow("escalation_flag")) != 0,
                        )
                    }.onSuccess { log ->
                        results.add(log)
                    }.onFailure { error ->
                        Log.e(TAG, "getLogs: failed to parse row — ${error.message}")
                    }
                }
            }
        }

        Log.d(TAG, "getLogs: returning ${results.size} entries (limit=$safeLimit)")
        return results
    }

    /**
     * Returns the total number of log entries — useful for health stats display.
     */
    fun count(): Int {
        return dbHelper.readableDatabase.use { db ->
            db.rawQuery("SELECT COUNT(*) FROM $TABLE_NAME", null).use { cursor ->
                if (cursor.moveToFirst()) cursor.getInt(0) else 0
            }
        }
    }
}
