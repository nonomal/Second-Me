"""
Migration: Add is_trained field to memory table
Version: 20250519144900
"""

description = "Add is_trained field to memory table"

def upgrade(conn):
    """
    Apply the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # Check if is_trained column already exists in memory table
    cursor.execute("PRAGMA table_info(memory)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Add is_trained field if it doesn't exist
    if 'is_trained' not in columns:
        cursor.execute("ALTER TABLE memory ADD COLUMN is_trained TEXT CHECK(is_trained IN ('yes', 'no')) NOT NULL DEFAULT 'no'")
        print("Added is_trained column to memory table")
    
    # No need to commit, the migration manager handles transactions

def downgrade(conn):
    """
    Revert the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # SQLite doesn't support dropping columns directly
    # We need to create a new table without the is_trained field, copy the data, and replace the old table
    
    # Create a temporary table without is_trained field
    cursor.execute("""
    CREATE TABLE memory_temp (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        document_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        -- Add other original columns here, but exclude is_trained
    )
    """)
    
    # Copy data to the temporary table (excluding is_trained field)
    cursor.execute("""
    INSERT INTO memory_temp (id, title, content, document_id, created_at, updated_at)
    SELECT id, title, content, document_id, created_at, updated_at FROM memory
    """)
    
    # Drop the original table
    cursor.execute("DROP TABLE memory")
    
    # Rename the temporary table to the original table name
    cursor.execute("ALTER TABLE memory_temp RENAME TO memory")
    
    print("Removed is_trained column from memory table")
