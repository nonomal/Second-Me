#!/bin/bash

# Set error handling
set -e

# Navigate to app directory
cd /app

# Set environment variables
echo "配置环境变量..."
export PYTHONPATH=/app:${PYTHONPATH}

# Load environment variables from .env file
set -a
source /app/.env
set +a

# Use mapped directories from docker-compose
export BASE_DIR=/app/data
export LOCAL_LOG_DIR=/app/logs
export RUN_DIR=/app/run
export RESOURCES_DIR=/app/resources

# Create log file if it doesn't exist
LOG_FILE="${LOCAL_LOG_DIR}/backend.log"
touch "$LOG_FILE"

# Start the application
echo "启动应用程序..."
echo "应用程序将在以下地址运行:"
echo "- 容器内访问: http://localhost:${LOCAL_APP_PORT:-8002}"
echo "- 主机访问: http://localhost:8002 (映射端口)"

# Log startup
echo "Starting at $(date)" >> "$LOG_FILE"
echo "Running: python -m lpm_kernel.main" >> "$LOG_FILE"

# Start the application and log output
exec python -m lpm_kernel.main >> "$LOG_FILE" 2>&1
