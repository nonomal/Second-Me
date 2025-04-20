from flask import Flask, request
from .common.repository.database_session import DatabaseSession, Base
from .common.logging import logger
from .api import init_routes
from .api.file_server.handler import FileServerHandler
from .database.migration_manager import MigrationManager
import os
import atexit


def create_app():
    app = Flask(__name__)

    # Initialize database connection
    try:
        DatabaseSession.initialize()
        logger.info("Database connection initialized successfully")
        
        # Run database migrations
        try:
            # Get database path and ensure directory exists
            db_path = os.getenv("SQLITE_DB_PATH", os.path.join(os.getenv("BASE_DIR", "data"), "sqlite", "lpm.db"))
            db_dir = os.path.dirname(db_path)
            
            # Ensure database directory exists
            if not os.path.exists(db_dir):
                logger.info(f"Creating database directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)
                
            migrations_dir = os.path.join(os.path.dirname(__file__), "database", "migrations")
            logger.info(f"Running migrations from: {migrations_dir} on database: {db_path}")
            
            # Apply any pending migrations
            manager = MigrationManager(db_path)
            applied = manager.apply_migrations(migrations_dir)
            
            if applied:
                logger.info(f"Successfully applied {len(applied)} database migrations: {', '.join(applied)}")
            else:
                logger.info("No new database migrations to apply")
                
        except Exception as migration_error:
            logger.error(f"Failed to run database migrations: {str(migration_error)}", exc_info=True)
            # Continue app startup even if migrations fail
            # This allows the app to start with existing schema
            
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {str(e)}")
        raise

        # Add CORS support

    @app.after_request
    def after_request(response):
        # Allow all origins in development environment
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type,Authorization"
        )
        response.headers.add(
            "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
        )
        return response

    # Create file server handler
    file_handler = FileServerHandler(
        os.path.join(os.getenv("APP_ROOT", "/app"), "resources", "raw_content")
    )

    @app.route("/raw_content/", defaults={"path": ""})
    @app.route("/raw_content/<path:path>")
    def serve_content(path=""):
        return file_handler.handle_request(path, request.path)

    # Register all routes
    init_routes(app)

    # Clean up database connection only when the application shuts down
    @app.teardown_appcontext
    def cleanup_db(exception):
        pass

    return app


app = create_app()


@atexit.register
def cleanup():
    DatabaseSession.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
