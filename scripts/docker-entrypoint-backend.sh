#!/bin/bash

# Set error handling
set -e

# Navigate to app directory
cd /app

# Set environment variables
echo "Setting environment variables..."
export PYTHONPATH=/app:${PYTHONPATH}

# Load environment variables from .env file
set -a
source /app/.env
set +a

# Use mapped directories from docker-compose
# ./data is mapped to /app/data
# ./logs is mapped to /app/logs
# ./run is mapped to /app/run
# ./resources is mapped to /app/resources
export BASE_DIR=/app/data
export LOCAL_LOG_DIR=/app/logs
export RUN_DIR=/app/run
export RESOURCES_DIR=/app/resources

# Flask application settings
export FLASK_APP=lpm_kernel.main
export FLASK_ENV=${FLASK_ENV:-production}

# Output Python information
echo "Checking Python environment..."
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"
PYTHON_VERSION=$(python --version)
echo "Python version: $PYTHON_VERSION"

# Check necessary Python packages
echo "Checking necessary Python packages..."
python -c "import flask" || { echo "Error: Missing flask package"; exit 1; }
python -c "import chromadb" || { echo "Error: Missing chromadb package"; exit 1; }

# Check if sqlite3 is installed
if ! command -v sqlite3 &> /dev/null; then
    echo "Installing sqlite3..."
    apt-get update && apt-get install -y sqlite3
fi

# Initialize database
echo "Initializing database..."
SQLITE_DB_PATH="${BASE_DIR}/sqlite/lpm.db"
mkdir -p "${BASE_DIR}/sqlite"

if [ ! -f "$SQLITE_DB_PATH" ]; then
    echo "Initializing database..."
    
    # Check if init.sql exists in different possible locations
    INIT_SQL_PATHS=(
        "/app/docker/sqlite/init.sql"
        "/app/resources/init.sql"
        "/app/sqlite/init.sql"
    )
    
    INIT_SQL_FOUND=false
    for path in "${INIT_SQL_PATHS[@]}"; do
        if [ -f "$path" ]; then
            echo "Found init.sql at: $path"
            cat "$path" | sqlite3 "$SQLITE_DB_PATH"
            INIT_SQL_FOUND=true
            break
        fi
    done
    
    if [ "$INIT_SQL_FOUND" = false ]; then
        echo "Warning: Could not find init.sql, creating empty database"
        touch "$SQLITE_DB_PATH"
    fi
    
    # Set default configurations
    echo "Setting default configurations..."
    python -c "from lpm_kernel.api.services.config_service import ConfigService; ConfigService().ensure_default_configs()"
    
    echo "Database initialization completed"
else
    echo "Database already exists at: $SQLITE_DB_PATH"
fi

# Ensure necessary directories exist
echo "Creating necessary directories..."
mkdir -p ${BASE_DIR}/chroma_db
mkdir -p ${LOCAL_LOG_DIR}
mkdir -p ${RUN_DIR}

# Initialize ChromaDB
echo "Initializing ChromaDB..."
INIT_CHROMA_PATHS=(
    "/app/docker/app/init_chroma.py"
    "/app/resources/init_chroma.py"
)

INIT_CHROMA_FOUND=false
for path in "${INIT_CHROMA_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "Found init_chroma.py at: $path"
        python "$path"
        INIT_CHROMA_FOUND=true
        break
    fi
done

if [ "$INIT_CHROMA_FOUND" = false ]; then
    echo "Warning: Could not find init_chroma.py"
fi

# Start Flask application
echo "Starting Flask application..."
echo "Application will run at the following addresses:"
echo "- Container access: http://localhost:${LOCAL_APP_PORT:-8002}"
echo "- Host access: http://localhost:8002 (mapped port)"

# Create log file if it doesn't exist
LOG_FILE="${LOCAL_LOG_DIR}/backend.log"
touch "$LOG_FILE"

# Start the application and log output
echo "Starting at $(date)" >> "$LOG_FILE"
exec python -m flask run --host=0.0.0.0 --port=${LOCAL_APP_PORT:-8002} >> "$LOG_FILE" 2>&1
