"""
Migration: Add cloud_service_api_key field to user_llm_configs table
Version: 20250521170000
"""

description = "Add cloud_service_api_key field to user_llm_configs table"

def upgrade(conn):
    """
    Apply the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # Check if cloud_service_api_key column already exists in user_llm_configs table
    cursor.execute("PRAGMA table_info(user_llm_configs)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Add cloud_service_api_key field if it doesn't exist
    if 'cloud_service_api_key' not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN cloud_service_api_key VARCHAR(200)")
        print("Added cloud_service_api_key column to user_llm_configs table")
    
    # No need to commit, the migration manager handles transactions
