#!/bin/bash

# Set error handling
set -e

# Source conda
source /opt/conda/etc/profile.d/conda.sh

# Activate conda environment
echo "Activating conda environment: $CONDA_DEFAULT_ENV"
conda activate $CONDA_DEFAULT_ENV

# Navigate to app directory
cd /app

# Check if we need to install/update dependencies
if [ ! -f "/app/.dependencies_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -e .
    
    # If frontend exists, install npm dependencies
    if [ -d "/app/lpm_frontend" ] && [ -f "/app/lpm_frontend/package.json" ]; then
        echo "Installing frontend dependencies..."
        cd /app/lpm_frontend
        npm install
        cd /app
    fi
    
    # Mark dependencies as installed
    touch /app/.dependencies_installed
fi

# Start backend service
echo "Starting Second-Me backend service..."
cd /app
python -m lpm_kernel.main &
BACKEND_PID=$!

# Wait for backend to start (adjust timeout as needed)
echo "Waiting for backend to become available..."
ATTEMPTS=0
MAX_ATTEMPTS=30
while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:$LOCAL_APP_PORT/health &>/dev/null; then
        echo "Backend service is up and running!"
        break
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    echo "Waiting for backend (attempt $ATTEMPTS/$MAX_ATTEMPTS)..."
    sleep 2
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
    echo "Backend failed to start within the expected time"
    exit 1
fi

# Start frontend service if it exists
if [ -d "/app/lpm_frontend" ] && [ -f "/app/lpm_frontend/package.json" ]; then
    echo "Starting frontend service..."
    cd /app/lpm_frontend
    npm run dev &
    FRONTEND_PID=$!
    cd /app
fi

# Function to handle termination
cleanup() {
    echo "Received termination signal. Shutting down services..."
    
    # Kill processes
    if [ -n "$FRONTEND_PID" ]; then
        kill -TERM $FRONTEND_PID 2>/dev/null || true
    fi
    
    if [ -n "$BACKEND_PID" ]; then
        kill -TERM $BACKEND_PID 2>/dev/null || true
    fi
    
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Keep container running
echo "All services started successfully. Container is now running..."
wait
