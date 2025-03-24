#!/bin/bash

# Script version
VERSION="1.0.0"

# Source conda utilities
SCRIPT_DIR="$( cd "$( dirname "${0:A}" )" && pwd )"
source "$SCRIPT_DIR/conda_utils.sh"

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

log_success() {
    echo -e "${GRAY}[$(get_timestamp)]${NC} ${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${GRAY}[$(get_timestamp)]${NC} ${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${GRAY}[$(get_timestamp)]${NC} ${RED}[ERROR]${NC}   $1"
}

log_section() {
    echo -e "${GRAY}[$(get_timestamp)]${NC} ${BOLD}[SECTION]${NC} $1"
}

# Check if port is available
check_port() {
    local port=$1
    if lsof -i:${port} > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Check if backend is healthy
check_backend_health() {
    local max_attempts=$1
    local attempt=1
    local backend_url="http://127.0.0.1:${LOCAL_APP_PORT}/health"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$backend_url" &>/dev/null; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    return 1
}

# Check if frontend is ready
check_frontend_ready() {
    local max_attempts=$1
    local attempt=1
    local frontend_log="../logs/frontend.log"
    
    while [ $attempt -le $max_attempts ]; do
        if grep -q "Local:" "$frontend_log" 2>/dev/null; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    return 1
}

# These functions are now sourced from conda_utils.sh

# Check if setup is complete
check_setup_complete() {
    log_info "Checking if setup is complete..."
    
    # Check conda environment
    if is_custom_conda_mode; then
        log_info "Custom conda mode enabled, verifying environment..."
        if ! verify_conda_env; then
            return 1
        fi
    else
        if ! conda env list | grep -q "${CONDA_DEFAULT_ENV:-second-me}"; then
            log_error "Conda environment '${CONDA_DEFAULT_ENV:-second-me}' not found. Please run 'make setup' first."
            return 1
        fi
    fi
    
    # Check if frontend dependencies are installed
    if [ ! -d "lpm_frontend/node_modules" ] && [ "$BACKEND_ONLY" != "true" ]; then
        log_error "Frontend dependencies not installed. Please run 'make setup' first."
        return 1
    fi
    
    log_success "Setup check passed"
    return 0
}

# Main function to start services
start_services() {
    log_section "STARTING SERVICES"
    
    # Parse arguments
    BACKEND_ONLY="false"
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --backend-only) BACKEND_ONLY="true" ;;
            *) log_error "Unknown parameter: $1"; return 1 ;;
        esac
        shift
    done
    
    # Check if setup is complete
    if ! check_setup_complete; then
        return 1
    fi
    
    # Load environment variables
    if [[ -f .env ]]; then
        export CONDA_DEFAULT_ENV="$(grep '^CONDA_DEFAULT_ENV=' .env | cut -d '=' -f2)"
        export LOCAL_APP_PORT="$(grep '^LOCAL_APP_PORT=' .env | cut -d '=' -f2)"
        export LOCAL_FRONTEND_PORT="$(grep '^LOCAL_FRONTEND_PORT=' .env | cut -d '=' -f2)"
        
        if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
            export CONDA_DEFAULT_ENV="second-me"
        fi
        if [[ -z "$LOCAL_APP_PORT" ]]; then
            export LOCAL_APP_PORT="8002"
        fi
        if [[ -z "$LOCAL_FRONTEND_PORT" ]]; then
            export LOCAL_FRONTEND_PORT="3000"
        fi
    else
        log_error ".env file not found!"
        return 1
    fi
    
    # Check if ports are available
    log_info "Checking port availability..."
    if ! check_port ${LOCAL_APP_PORT}; then
        log_error "Backend port ${LOCAL_APP_PORT} is already in use!"
        return 1
    fi
    if ! check_port ${LOCAL_FRONTEND_PORT} && [[ "$BACKEND_ONLY" != "true" ]]; then
        log_error "Frontend port ${LOCAL_FRONTEND_PORT} is already in use!"
        return 1
    fi
    log_success "All ports are available"
    
    # Initialize conda environment if not using custom configuration
    if ! is_custom_conda_mode; then
        log_info "Initializing conda environment..."
        local conda_cmd
        if ! conda_cmd=$(try_source_conda_sh_all); then
            log_error "Failed to initialize conda environment"
            return 1
        fi
        
        # get conda.sh path
        local conda_sh_path
        if [[ -n "$conda_cmd" ]]; then
            local conda_root="$(dirname "$(dirname "$conda_cmd")")"
            if ! find_and_source_conda_sh "$conda_root" conda_sh_path; then
                log_error "Could not find conda.sh activation script"
                return 1
            fi
        else
            log_error "conda command not found"
            return 1
        fi
        log_info "Found conda.sh at: $conda_sh_path"
        log_success "Conda initialized successfully: $conda_cmd"
    else
        log_info "Using custom conda environment, skipping conda initialization"
        conda_sh_path="" # Set empty value since it won't be used
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    mkdir -p run
    
    # Start backend service
    log_info "Starting backend service..."
    
    # Check if using custom conda mode
    if is_custom_conda_mode; then
        log_info "Using custom conda environment, skipping conda activation"
        nohup zsh -c ./scripts/start_local.sh > logs/start.log 2>&1 &
    else
        log_info "Using conda environment: ${CONDA_DEFAULT_ENV}"
        # Ensure conda is properly initialized in the subshell
        nohup zsh -c "source ${conda_sh_path} && conda activate ${CONDA_DEFAULT_ENV} && exec ./scripts/start_local.sh" > logs/start.log 2>&1 &
    fi
    
    echo $! > run/.backend.pid
    log_info "Backend service started in background with PID: $(cat run/.backend.pid)"
    
    # Wait for backend to be healthy
    log_info "Waiting for backend service to be ready..."
    if ! check_backend_health 300; then
        log_error "Backend service failed to start within 300 seconds"
        return 1
    fi
    log_success "Backend service is ready"
    
    # Start frontend service if not backend-only mode
    if [[ "$BACKEND_ONLY" != "true" ]]; then
        if [ ! -d "lpm_frontend" ]; then
            log_error "Frontend directory 'lpm_frontend' not found!"
            return 1
        fi
        
        log_info "Starting frontend service..."
        cd lpm_frontend
        
        # Check if node_modules exists
        if [ ! -d "node_modules" ]; then
            log_info "Installing frontend dependencies..."
            if ! npm install; then
                log_error "Failed to install frontend dependencies"
                cd ..
                return 1
            fi
            log_success "Frontend dependencies installed"
        fi
        
        # Start frontend in background
        log_info "Starting frontend dev server..."
        nohup npm run dev > ../logs/frontend.log 2>&1 &
        echo $! > ../run/.frontend.pid
        log_info "Frontend service started in background with PID: $(cat ../run/.frontend.pid)"
        
        # Wait for frontend to be ready
        log_info "Waiting for frontend service to be ready..."
        if ! check_frontend_ready 300; then
            log_error "Frontend service failed to start within 300 seconds"
            cd ..
            return 1
        fi
        log_success "Frontend service is ready"
        cd ..
    fi
    
    # Display service URLs
    log_section "Services are running"
    if [[ "$BACKEND_ONLY" == "true" ]]; then
        log_info "Backend URL:  http://localhost:${LOCAL_APP_PORT}"
    else
        log_info "Frontend URL: http://localhost:${LOCAL_FRONTEND_PORT}"
        log_info "Backend URL:  http://localhost:${LOCAL_APP_PORT}"
    fi
}

# Execute start services
start_services "$@"

# Error handling
if [[ $? -ne 0 ]]; then
    log_error "Failed to start services!"
    exit 1
fi
