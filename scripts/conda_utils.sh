#!/bin/bash

# This file contains utility functions for conda initialization and management
# It is used by both setup.sh and start.sh to avoid code duplication

# Color definitions (if not already defined)
if [ -z "$RED" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    GRAY='\033[0;90m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
fi

# Logging functions (if not already defined)
if ! type log_info > /dev/null 2>&1; then
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
fi

# Find and source conda.sh script from a specified conda root or common locations
find_and_source_conda_sh() {
    local conda_root="$1"
    local return_path="$2"
    
    # If conda_root is provided, try to find conda.sh directly
    if [[ -n "$conda_root" ]]; then
        local conda_sh="$conda_root/etc/profile.d/conda.sh"
        if [[ -f "$conda_sh" ]]; then
            log_info "Found conda.sh script at $conda_sh"
            source "$conda_sh"
            
            # If a return path variable is provided, store the path in that variable
            if [[ -n "$return_path" ]]; then
                eval "$return_path=\"$conda_sh\""
            fi
            
            return 0
        else
            log_info "conda.sh not found at specified location: $conda_sh"
            log_info "Trying common locations..."
        fi
    fi
    
    # Try common locations as fallback
    local conda_sh_paths=(
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda/etc/profile.d/conda.sh"
        "$HOME/anaconda/etc/profile.d/conda.sh"
        "$HOME/opt/anaconda3/etc/profile.d/conda.sh"
        "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
        "/usr/local/Caskroom/anaconda/base/etc/profile.d/conda.sh"
        "/usr/share/miniconda3/etc/profile.d/conda.sh"
        "/usr/share/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/share/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/share/anaconda3/etc/profile.d/conda.sh"
    )
    
    local found=false
    for conda_sh_path in "${conda_sh_paths[@]}"; do
        if [[ -f "$conda_sh_path" ]]; then
            log_info "Found conda.sh script at $conda_sh_path"
            source "$conda_sh_path"
            found=true
            
            # If a return path variable is provided, store the path in that variable
            if [[ -n "$return_path" ]]; then
                eval "$return_path=\"$conda_sh_path\""
            fi
            
            break
        fi
    done

    if [[ "$found" = false ]]; then
        log_error "Could not find conda.sh script in any common locations"
        return 1
    fi
    
    log_info "[CONDA READY] Successfully initialized conda from common locations: $conda_sh_path"
    return 0
}

# Verify if conda command is working properly
verify_conda_command() {
    local conda_path="$1"
    
    # Try to run conda --version
    if ! "$conda_path" --version &>/dev/null; then
        return 1
    fi
    
    return 0
}

# Try to use a conda executable and initialize it
try_source_conda_sh_by_conda_executable() {
    local conda_path="$1"
    local source_description="$2"
    
    if [[ ! -f "$conda_path" ]]; then
        log_info "[CONDA CHECK] Checking $source_description: $conda_path (not found)"
        return 1
    fi
    
    log_info "[CONDA CHECK] Checking $source_description: $conda_path (found, verifying...)"
    
    # Determine conda root directory from executable location
    local conda_root="$(dirname "$(dirname "$conda_path")")"
    
    # Try to find and source conda.sh
    if ! find_and_source_conda_sh "$conda_root"; then
        log_error "[CONDA INIT FAILED] Found conda at $conda_path but could not find conda.sh"
        return 1
    fi
    
    # Verify conda is working
    if ! verify_conda_command "$conda_path"; then
        log_error "[CONDA VERIFICATION FAILED] Found conda at $conda_path but command is not working"
        return 1
    fi
    
    log_success "[CONDA READY] Successfully initialized conda from $source_description: $conda_path"
    echo "$conda_path"
    return 0
}

# Check for conda command and initialize it
try_source_conda_sh_all() {
    local conda_cmd
    
    # METHOD 1: Check if CONDA_EXE environment variable is set
    if [[ -n "$CONDA_EXE" ]]; then
        if conda_cmd=$(try_source_conda_sh_by_conda_executable "$CONDA_EXE" "CONDA_EXE environment variable"); then
            log_info "[CONDA READY] Successfully initialized conda from CONDA_EXE environment variable: $CONDA_EXE"
            return 0
        fi
    fi
    
    # METHOD 2: Check if conda executable is in PATH
    local conda_in_path="$(command -v conda 2>/dev/null)"
    if [[ -n "$conda_in_path" ]]; then
        if conda_cmd=$(try_source_conda_sh_by_conda_executable "$conda_in_path" "PATH"); then
            log_info "[CONDA READY] Successfully initialized conda from PATH: $conda_in_path"
            return 0
        fi
    fi
    
    # METHOD 3: Check common locations for conda executable
    local common_conda_paths=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniconda/bin/conda"
        "$HOME/anaconda/bin/conda"
        "$HOME/opt/anaconda3/bin/conda"
        "$HOME/opt/miniconda3/bin/conda"
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
        "/opt/miniconda3/bin/conda"
        "/opt/anaconda3/bin/conda"
        "/usr/local/miniconda3/bin/conda"
        "/usr/local/anaconda3/bin/conda"
        "/usr/local/Caskroom/miniconda/base/bin/conda"
        "/usr/local/Caskroom/anaconda/base/bin/conda"
        "/usr/share/miniconda3/bin/conda"
        "/usr/share/anaconda3/bin/conda"
        "/usr/local/share/miniconda3/bin/conda"
        "/usr/local/share/anaconda3/bin/conda"
    )
    
    # Search through common locations for conda executable
    for conda_path in "${common_conda_paths[@]}"; do
        if conda_cmd=$(try_source_conda_sh_by_conda_executable "$conda_path" "common locations"); then
            log_info "[CONDA READY] Successfully initialized conda from common locations: $conda_path"
            return 0
        fi
    done
    
    # Last resort: Try to find conda.sh directly without finding conda executable first
    log_info "[LAST RESORT] Trying to find conda.sh directly from common locations"
    if find_and_source_conda_sh "/opt/homebrew/Caskroom/miniconda/base"; then
        # If conda.sh was found and sourced, conda command should be available
        if command -v conda &>/dev/null; then
            log_info "[CONDA COMMAND AVAILABLE] After sourcing conda.sh, conda command is available"
            echo "conda"
            return 0
        fi
    fi
    
    # If we reach here, we couldn't find conda executable or conda.sh anywhere
    log_error "[CONDA EXECUTABLE NOT FOUND] conda executable not found in PATH or common locations"
    return 1
}

# Check if custom conda mode is enabled
is_custom_conda_mode() {
    if [ -f ".env" ]; then
        local custom_mode=$(grep '^CUSTOM_CONDA_MODE=' .env | cut -d '=' -f2)
        if [ "$custom_mode" = "true" ]; then
            log_warning "Custom conda mode is enabled"
            return 0
        fi
    fi
    return 1
}

# Verify if current conda environment matches the required one
verify_conda_env() {
    if [ -f ".env" ]; then
        local env_name=$(grep '^CONDA_DEFAULT_ENV=' .env | cut -d '=' -f2)
        if [ -n "$env_name" ]; then
            log_info "Verifying conda environment..."
            
            # Verify current environment exists in system
            local current_env=$(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}')
            if [ "$current_env" != "$env_name" ]; then
                log_warning "Please activate the specified conda environment: conda activate $env_name"
                log_info "Current environment: $current_env"
                return 1
            fi
            
            log_success "Conda environment verified: $env_name"
            return 0
        fi
    fi
    
    log_error "CONDA_DEFAULT_ENV not set in .env file"
    return 1
}

# Initialize conda environment
initialize_conda() {
    log_info "Initializing conda environment..."
    
    # Use our optimized try_source_conda_sh_all function to find and initialize conda
    local conda_cmd
    if ! conda_cmd=$(try_source_conda_sh_all); then
        log_error "Failed to initialize conda environment"
        return 1
    fi
    
    # Run conda init to ensure shell integration is properly set up
    log_info "Running conda init to set up shell integration..."
    if ! conda init zsh bash; then
        log_warning "Conda init may have encountered issues, but we'll try to continue"
    else
        log_success "Conda init completed successfully"
    fi
    
    # Source conda.sh to apply changes immediately
    log_info "Sourcing conda.sh to apply changes immediately..."
    
    # Determine conda root directory from conda command
    local conda_root=""
    if [ -n "$conda_cmd" ]; then
        conda_root=$(dirname $(dirname "$conda_cmd"))
        log_info "Determined conda root directory: $conda_root"
    fi
    
    local conda_sh_path
    if find_and_source_conda_sh "$conda_root" conda_sh_path; then
        log_success "Successfully sourced conda.sh from: $conda_sh_path"
    else
        log_warning "Could not source conda.sh, changes will apply after shell restart"
    fi
    
    log_success "Conda initialized successfully: $conda_cmd"
    return 0
}
