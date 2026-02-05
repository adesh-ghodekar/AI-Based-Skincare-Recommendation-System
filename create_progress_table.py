import sqlite3

CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    image_path TEXT,
    skin_type TEXT,
    conditions TEXT,
    routine TEXT,
    timestamp TEXT
);