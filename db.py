import json
import os
import mysql.connector
from mysql.connector import Error as MySQLError

# ─────────────────────────────────────────
# CONNECTION CONFIG
# ─────────────────────────────────────────
from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}



def get_connection():
    """Return a fresh MySQL connection."""
    return mysql.connector.connect(**MYSQL_CONFIG)


# ─────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────
def init_db():
    """
    Create the database (if missing) and both tables.
    Returns (success: bool, message: str).
    """
    cfg_no_db = {k: v for k, v in MYSQL_CONFIG.items() if k != "database"}
    try:
        # Step 1 — create database
        conn = mysql.connector.connect(**cfg_no_db)
        cur  = conn.cursor()
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        cur.close()
        conn.close()

        # Step 2 — create tables
        conn2 = get_connection()
        cur2  = conn2.cursor()

        cur2.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                meeting_title VARCHAR(255),
                meeting_date  DATE,
                attendees     TEXT,
                raw_text      LONGTEXT NOT NULL,
                cleaned_text  LONGTEXT,
                word_count    INT,
                speakers      TEXT,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cur2.execute("""
            CREATE TABLE IF NOT EXISTS meeting_minutes (
                id                INT AUTO_INCREMENT PRIMARY KEY,
                transcript_id     INT NOT NULL,
                summary           TEXT,
                topics            JSON,
                key_points        JSON,
                decisions         JSON,
                action_items      JSON,
                next_steps        JSON,
                overall_sentiment VARCHAR(20),
                meeting_type      VARCHAR(50),
                model_used        VARCHAR(80),
                created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcript_id)
                    REFERENCES transcripts(id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        conn2.commit()
        cur2.close()
        conn2.close()
        return True, "Database & tables ready."

    except MySQLError as e:
        return False, str(e)


# ─────────────────────────────────────────
# WRITE OPERATIONS
# ─────────────────────────────────────────
def save_transcript(title, date, attendees, raw_text, cleaned_text, word_count, speakers):
    """
    Insert a transcript row.
    Returns (new_id: int | None, error: str | None).
    """
    sql = """
        INSERT INTO transcripts
            (meeting_title, meeting_date, attendees,
             raw_text, cleaned_text, word_count, speakers)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (
            title or "Untitled",
            str(date),
            attendees or "",
            raw_text,
            cleaned_text,
            word_count,
            json.dumps(speakers),
        ))
        conn.commit()
        new_id = cur.lastrowid
        cur.close()
        conn.close()
        return new_id, None
    except MySQLError as e:
        return None, str(e)


def save_minutes(transcript_id, data, model_used):
    """
    Insert a meeting_minutes row linked to transcript_id.
    Returns (success: bool, error: str | None).
    """
    sql = """
        INSERT INTO meeting_minutes
            (transcript_id, summary, topics, key_points, decisions,
             action_items, next_steps, overall_sentiment, meeting_type, model_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (
            transcript_id,
            data.get("summary", ""),
            json.dumps(data.get("topics", [])),
            json.dumps(data.get("keyPoints", [])),
            json.dumps(data.get("decisions", [])),
            json.dumps(data.get("actionItems", [])),
            json.dumps(data.get("nextSteps", [])),
            data.get("overallSentiment", "neutral"),
            data.get("meetingType", "other"),
            model_used,
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True, None
    except MySQLError as e:
        return False, str(e)


# ─────────────────────────────────────────
# READ OPERATIONS
# ─────────────────────────────────────────
def fetch_past_transcripts(limit=5, search_term=None):
    """
    Return recent transcripts (optionally filtered by keyword).
    Each row: {id, meeting_title, meeting_date, cleaned_text,
               speakers, summary, meeting_type, overall_sentiment}
    Returns (rows: list[dict], error: str | None).
    """
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)

        if search_term:
            sql = """
                SELECT t.id, t.meeting_title, t.meeting_date,
                       t.cleaned_text, t.speakers,
                       m.summary, m.meeting_type, m.overall_sentiment
                FROM   transcripts t
                LEFT JOIN meeting_minutes m ON m.transcript_id = t.id
                WHERE  t.raw_text      LIKE %s
                   OR  t.meeting_title LIKE %s
                ORDER  BY t.created_at DESC
                LIMIT  %s
            """
            like = f"%{search_term}%"
            cur.execute(sql, (like, like, limit))
        else:
            sql = """
                SELECT t.id, t.meeting_title, t.meeting_date,
                       t.cleaned_text, t.speakers,
                       m.summary, m.meeting_type, m.overall_sentiment
                FROM   transcripts t
                LEFT JOIN meeting_minutes m ON m.transcript_id = t.id
                ORDER  BY t.created_at DESC
                LIMIT  %s
            """
            cur.execute(sql, (limit,))

        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows, None
    except MySQLError as e:
        return [], str(e)


def count_transcripts():
    """Return total number of stored transcripts (0 on error)."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transcripts")
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n
    except MySQLError:
        return 0


# ─────────────────────────────────────────
# DELETE OPERATIONS
# ─────────────────────────────────────────
def delete_transcript(transcript_id):
    """
    Delete a transcript (cascades to meeting_minutes).
    Returns (success: bool, error: str | None).
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("DELETE FROM transcripts WHERE id = %s", (transcript_id,))
        conn.commit()
        cur.close()
        conn.close()
        return True, None
    except MySQLError as e:
        return False, str(e)