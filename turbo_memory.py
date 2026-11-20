
import sqlite3

DATABASE = "turbo_memory.db"


# 1. Connect to TURBO's memory database
def connect_db():
    return sqlite3.connect(DATABASE)


# 2. Create the memory table
def initialize_memory():
    with connect_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                detail TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


# 3. Save a memory
def save_memory(category, detail):
    category = category.strip().lower()
    detail = detail.strip()

    if not category or not detail:
        return "Please provide a category and a memory."

    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO memories (category, detail)
            VALUES (?, ?)
            """,
            (category, detail)
        )

    return f"I saved that to memory: {detail}"


# 4. Recall memories
def recall_memories(search_term=None):
    with connect_db() as connection:
        if search_term:
            results = connection.execute(
                """
                SELECT id, category, detail, created_at
                FROM memories
                WHERE category LIKE ? OR detail LIKE ?
                ORDER BY id DESC
                """,
                (
                    f"%{search_term}%",
                    f"%{search_term}%"
                )
            ).fetchall()
        else:
            results = connection.execute(
                """
                SELECT id, category, detail, created_at
                FROM memories
                ORDER BY id DESC
                """
            ).fetchall()

    return results


# 5. Forget a memory by its ID
def forget_memory(memory_id):
    with connect_db() as connection:
        cursor = connection.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,)
        )

        deleted = cursor.rowcount

    if deleted:
        return f"Memory {memory_id} deleted."
    return f"I couldn't find memory {memory_id}."


# 6. Initialize the database when this file is imported
initialize_memory()


# 7. Run a simple test when launched directly
if __name__ == "__main__":
    print("TURBO memory system ready.")

    while True:
        print("\n1. Save memory")
        print("2. Recall memories")
        print("3. Search memories")
        print("4. Forget a memory")
        print("5. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            category = input("Category (personal, project, preference): ")
            detail = input("What should TURBO remember? ")
            print(save_memory(category, detail))

        elif choice == "2":
            memories = recall_memories()

            if memories:
                for item in memories:
                    print(
                        f"ID: {item[0]} | "
                        f"Category: {item[1]} | "
                        f"Memory: {item[2]} | "
                        f"Saved: {item[3]}"
                    )
            else:
                print("No memories saved yet.")

        elif choice == "3":
            term = input("What do you want to search for? ")
            memories = recall_memories(term)

            if memories:
                for item in memories:
                    print(f"ID: {item[0]} | {item[1]}: {item[2]}")
            else:
                print("No matching memories found.")

        elif choice == "4":
            try:
                memory_id = int(input("Enter the memory ID to forget: "))
                print(forget_memory(memory_id))
            except ValueError:
                print("Please enter a valid numeric ID.")

        elif choice == "5":
            print("TURBO memory system closed.")
            break

        else:
            print("Invalid choice. Please choose 1–5.")
