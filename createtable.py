import sqlite3

conn = sqlite3.connect('fiora_ai.db')
cursor = conn.cursor()

# Add the 'image' column if it doesn't exist
try:
    cursor.execute("ALTER TABLE progress ADD COLUMN image TEXT")
    print("✅ 'image' column added successfully.")
except sqlite3.OperationalError as e:
    print("⚠️ Column may already exist or another error:", e)

conn.commit()
conn.close()