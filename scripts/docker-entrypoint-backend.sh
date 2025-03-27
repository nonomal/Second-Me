#!/bin/bash

# Set error handling
set -e

# Navigate to app directory
cd /app

# Start backend service
echo "Starting Second-Me backend service..."
python -m lpm_kernel.main &
BACKEND_PID=$!

# Wait for backend to become available (health check)
echo "Waiting for backend to initialize..."
ATTEMPTS=0
MAX_ATTEMPTS=30
while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:${LOCAL_APP_PORT:-8002}/health &>/dev/null; then
        echo "Backend service is up and running!"
        break
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    echo "Waiting for backend initialization (attempt $ATTEMPTS/$MAX_ATTEMPTS)..."
    sleep 2
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
    echo "Backend failed to initialize within the expected time"
    exit 1
fi

echo "Backend service started successfully. Container is now running..."

# Keep container running
wait
