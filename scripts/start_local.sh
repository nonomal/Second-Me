#!/bin/bash

# Set environment variables
echo "Setting environment variables..."
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Load environment variables from .env file
set -a
source .env
set +a

# Use local base directory
export BASE_DIR=${LOCAL_BASE_DIR}

# Ensure using the correct Python environment
echo "Checking Python environment..."
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"
PYTHON_VERSION=$(python --version)
echo "Python version: $PYTHON_VERSION"
CONDA_ENV=$(echo $CONDA_DEFAULT_ENV)
echo "Conda environment: $CONDA_ENV"

# Check necessary Python packages
echo "Checking necessary Python packages..."
python -c "import flask" || { echo "Error: Missing flask package"; exit 1; }
python -c "import chromadb" || { echo "Error: Missing chromadb package"; exit 1; }

# Initialize database
echo "Initializing database..."
SQLITE_DB_PATH="${BASE_DIR}/data/sqlite/lpm.db"
mkdir -p "${BASE_DIR}/data/sqlite"

if [ ! -f "$SQLITE_DB_PATH" ]; then
    echo "Initializing database..."
    cat docker/sqlite/init.sql | sqlite3 "$SQLITE_DB_PATH"
    echo "Database initialization completed"
else
    echo "Database already exists"
fi

# Ensure necessary directories exist
echo "Checking necessary directories..."
mkdir -p ${BASE_DIR}/data/chroma_db
mkdir -p ${LOCAL_LOG_DIR}
#mkdir -p ${BASE_DIR}/raw_content
#mkdir -p ${BASE_DIR}/data_pipeline

# Initialize ChromaDB
echo "Initializing ChromaDB..."
python docker/app/init_chroma.py

# Get local IP address (excluding localhost and docker networks)
LOCAL_IP=$(ifconfig | grep "inet " | grep -v "127.0.0.1" | grep "192.168" | awk '{print $2}' | head -n 1)

# Start Flask application
echo "Starting Flask application..."
echo "Application will run at the following addresses:"
echo "- Local access: http://localhost:${LOCAL_APP_PORT}"
echo "- LAN access: http://${LOCAL_IP}:${LOCAL_APP_PORT}"

# Output logs to file
exec python -m flask run --host=0.0.0.0 --port=${LOCAL_APP_PORT} >> "${LOCAL_LOG_DIR}/backend.log" 2>&1
