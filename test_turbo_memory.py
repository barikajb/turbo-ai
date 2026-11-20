
import sqlite3

# Create or open TURBO's local memory database
connection = sqlite3.connect("turbo_memory.db")

cursor = connection.cursor()

# Create a table for saved memories
cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        detail TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

connection.commit()

print("TURBO memory database created successfully.")

# Save a test memory
cursor.execute(
    "INSERT INTO memories (category, detail) VALUES (?, ?)",
    ("test", "TURBO memory is working")
)

connection.commit()

# Read saved memories
cursor.execute("SELECT id, category, detail FROM memories")

for memory in cursor.fetchall():
    print(memory)

connection.close()
