#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

check_backend_health() {
    local max_attempts=$1
    local attempt=1
    local backend_url="http://127.0.0.1:${LOCAL_APP_PORT}/health"
    local backend_log="logs/start.log"
    local log_pid=0

    log_info "Waiting for backend service to be ready (showing real-time logs)..."

    # Start real-time log display in background if log file exists
    if [ -f "$backend_log" ]; then
        echo -e "${GRAY}---Backend logs begin (real-time)---${NC}"
        tail -f "$backend_log" &
        log_pid=$!
    fi

    while [ $attempt -le $max_attempts ]; do
        # Non-blocking health check
        if curl -s -f "$backend_url" &>/dev/null; then
            # Stop the log display process
            if [ $log_pid -ne 0 ]; then
                kill $log_pid >/dev/null || true
                echo -e "${GRAY}---Backend logs end---${NC}"
            fi
            return 0
        fi

        sleep 1
        attempt=$((attempt + 1))
    done

    # Stop the log display process if it's still running
    if [ $log_pid -ne 0 ]; then
        kill $log_pid >/dev/null || true
        echo -e "${GRAY}---Backend logs end---${NC}"
    fi

    return 1
}
if ! check_backend_health 300; then
    log_error "Backend service failed to start within 300 seconds"
    exit 1
fi
log_success "Service is ready"