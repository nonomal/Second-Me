"""
Database Migration Manager

This module provides functionality to manage database migrations in a systematic way.
It ensures that migrations are applied in order and only once.
"""

import os
import sys
import importlib.util
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MigrationManager:
    """Manages database migrations for SQLite database"""
    
    def __init__(self, db_path):
        """
        Initialize the migration manager
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create migration tracking table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(50) PRIMARY KEY,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        conn.commit()
        conn.close()
        logger.debug("Migration tracking table checked/created")
    
    def get_applied_migrations(self):
        """
        Get list of already applied migrations
        
        Returns:
            List of applied migration versions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        versions = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return versions
    
    def apply_migrations(self, migrations_dir=None):
        """
        Apply all pending migrations from the migrations directory
        
        Args:
            migrations_dir: Directory containing migration scripts.
                            If None, use 'migrations' subdirectory
        
        Returns:
            List of applied migration versions
        """
        if migrations_dir is None:
            migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
        
        # Ensure migrations directory exists
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Get already applied migrations
        applied = self.get_applied_migrations()
        logger.info(f"Found {len(applied)} previously applied migrations")
        
        # Get all migration files and sort them
        migration_files = []
        for f in os.listdir(migrations_dir):
            if f.endswith('.py') and not f.startswith('__'):
                try:
                    # Extract version from filename (format: V1_2_3__description.py)
                    version = f.split('__')[0].replace('V', '')
                    migration_files.append((version, f))
                except Exception as e:
                    logger.warning(f"Skipping invalid migration filename: {f}, error: {e}")
        
        # Sort by version
        migration_files.sort(key=lambda x: x[0])
        
        applied_in_session = []
        
        # Apply each migration that hasn't been applied yet
        for version, migration_file in migration_files:
            if version in applied:
                logger.debug(f"Skipping already applied migration: {migration_file}")
                continue
            
            logger.info(f"Applying migration: {migration_file}")
            
            # Load the migration module
            module_path = os.path.join(migrations_dir, migration_file)
            module_name = f"migration_{version}"
            
            try:
                # Import the migration module dynamically
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                migration_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(migration_module)
                
                # Get migration description
                description = getattr(migration_module, 'description', migration_file)
                
                # Connect to database and start transaction
                conn = sqlite3.connect(self.db_path)
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    # Execute the migration
                    migration_module.upgrade(conn)
                    
                    # Record the migration
                    conn.execute(
                        "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                        (version, description)
                    )
                    
                    # Commit the transaction
                    conn.commit()
                    logger.info(f"Successfully applied migration: {migration_file}")
                    applied_in_session.append(version)
                    
                except Exception as e:
                    # Rollback on error
                    conn.rollback()
                    logger.error(f"Error applying migration {migration_file}: {str(e)}")
                    raise
                
                finally:
                    conn.close()
                
            except Exception as e:
                logger.error(f"Failed to load migration {migration_file}: {str(e)}")
                raise
        
        if not applied_in_session:
            logger.info("No new migrations to apply")
        else:
            logger.info(f"Applied {len(applied_in_session)} new migrations")
        
        return applied_in_session
    
    def create_migration(self, description, migrations_dir=None):
        """
        Create a new migration file with template code
        
        Args:
            description: Short description of what the migration does
            migrations_dir: Directory to create migration in
        
        Returns:
            Path to the created migration file
        """
        if migrations_dir is None:
            migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
        
        # Ensure migrations directory exists
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Get current timestamp for version
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Format description for filename (lowercase, underscores)
        safe_description = description.lower().replace(' ', '_').replace('-', '_')
        safe_description = ''.join(c for c in safe_description if c.isalnum() or c == '_')
        
        # Create filename
        filename = f"V{timestamp}__{safe_description}.py"
        filepath = os.path.join(migrations_dir, filename)
        
        # Create migration file with template
        with open(filepath, 'w') as f:
            f.write(f'''"""
Migration: {description}
Version: {timestamp}
"""

description = "{description}"

def upgrade(conn):
    """
    Apply the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # TODO: Implement your migration logic here
    # Example:
    # cursor.execute("""
    #     CREATE TABLE IF NOT EXISTS new_table (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         name TEXT NOT NULL
    #     )
    # """)
    
    # No need to commit, the migration manager handles transactions

def downgrade(conn):
    """
    Revert the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # TODO: Implement your downgrade logic here
    # Example:
    # cursor.execute("DROP TABLE IF EXISTS new_table")
    
    # No need to commit, the migration manager handles transactions
''')
        
        logger.info(f"Created new migration: {filename}")
        return filepath
