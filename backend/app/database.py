"""
database.py
───────────
Sets up SQLite database with 3 tables:
  - users       → stores registered accounts
  - uploads     → tracks every uploaded dataset
  - predictions → stores all AI prediction results
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'quantai.db')

def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn

def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── Users table ────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name  TEXT NOT NULL,
            last_name   TEXT NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ── Uploads table ──────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT UNIQUE NOT NULL,
            filename    TEXT NOT NULL,
            file_rows   INTEGER,
            file_cols   INTEGER,
            user_email  TEXT,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ── Predictions table ──────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            type         TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES uploads(session_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialised at:", os.path.abspath(DB_PATH))
