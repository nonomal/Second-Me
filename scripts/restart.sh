#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get current timestamp
get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Print formatted log messages
log_info() {
    echo -e "${GRAY}[$(get_timestamp)]${NC} ${GREEN}[INFO]${NC}    $1"
}

log_section() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}\n"
}

# Main function to restart services
restart_services() {
    log_section "RESTARTING SERVICES"
    
    # Stop services
    log_info "Stopping services..."
    ./scripts/stop.sh
    
    # Start services
    log_info "Starting services..."
    ./scripts/start.sh
}

# Execute restart services
restart_services
