import sqlite3
from datetime import datetime

DATABASE = 'fiora_ai.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS skin_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                skin_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                concerns TEXT,
                age_group TEXT,
                budget TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            
            CREATE TABLE IF NOT EXISTS routines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_ids TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        ''')

def authenticate_user(username, password):
    with get_db() as conn:
        user = conn.execute(
            'SELECT id, username FROM users WHERE username = ? AND password = ?',
            (username, password)
        ).fetchone()
    return dict(user) if user else None

def save_analysis(user_id, image_path, skin_type, confidence, concerns, age_group, budget):
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO skin_analysis 
               (user_id, image_path, skin_type, confidence, concerns, age_group, budget)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (user_id, image_path, skin_type, confidence, str(concerns), age_group, budget)
        )

def get_user_history(user_id):
    with get_db() as conn:
        analyses = conn.execute(
            '''SELECT * FROM skin_analysis 
               WHERE user_id = ? 
               ORDER BY created_at DESC''',
            (user_id,)
        ).fetchall()
    return [dict(a) for a in analyses]

def save_routine(user_id, product_ids):
    with get_db() as conn:
        conn.execute(
            'INSERT INTO routines (user_id, product_ids) VALUES (?, ?)',
            (user_id, ','.join(product_ids))
        )

def get_routine(user_id):
    with get_db() as conn:
        routine = conn.execute(
            'SELECT product_ids FROM routines WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
            (user_id,)
        ).fetchone()
    return routine['product_ids'].split(',') if routine else []

def save_review(user_id, rating, comment):
    with get_db() as conn:
        conn.execute(
            'INSERT INTO reviews (user_id, rating, comment) VALUES (?, ?, ?)',
            (user_id, rating, comment)
        )

# Initialize database on first run
init_db()
