"""
Migration: Add thinking models table
Version: 20250418154200
"""

description = "Add thinking models table and insert default model"

def upgrade(conn):
    """
    Apply the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # Check if thinking_models table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='thinking_models'")
    if cursor.fetchone():
        print("thinking_models table already exists, skipping creation")
        return
    
    # Create thinking_models table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS thinking_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thinking_model_name VARCHAR(200) NOT NULL,
        thinking_endpoint VARCHAR(200) NOT NULL,
        thinking_api_key VARCHAR(200),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create index for created_at
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_thinking_models_created_at ON thinking_models(created_at)")
    
    # Insert default thinking model
    cursor.execute("""
    INSERT INTO thinking_models (
        thinking_model_name, 
        thinking_endpoint, 
        thinking_api_key
    ) VALUES (?, ?, ?)
    """, ("gpt-4o", "https://api.openai.com/v1", ""))
    
    # No need to commit, the migration manager handles transactions

def downgrade(conn):
    """
    Revert the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # Drop the thinking_models table
    cursor.execute("DROP TABLE IF EXISTS thinking_models")
    
    # No need to commit, the migration manager handles transactions
