"""
Database manager for Disease Prediction System
Stores screening history using SQLite.
"""

import sqlite3
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(__file__), "disease_prediction.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_name TEXT,
            disease_type TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            prediction TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(patient_name, disease_type, input_summary, prediction, risk_level, confidence):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO prediction_history (timestamp, patient_name, disease_type, input_summary, prediction, risk_level, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, patient_name or "Anonymous", disease_type, input_summary, prediction, risk_level, confidence))
    conn.commit()
    conn.close()

def get_history(limit=50):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, patient_name, disease_type, input_summary, prediction, risk_level, confidence
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def clear_history():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
