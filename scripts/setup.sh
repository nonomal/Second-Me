#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Version
VERSION="1.0.0"

# Source conda utilities
SCRIPT_DIR="$( cd "$( dirname "${0:A}" )" && pwd )"
source "$SCRIPT_DIR/conda_utils.sh"

# Total number of stages
TOTAL_STAGES=6
CURRENT_STAGE=0
STAGE_NAME=""

# Trap ctrl-c and call cleanup
trap cleanup INT

# Cleanup function to restore terminal settings
cleanup() {
    echo -e "\n${YELLOW}Setup interrupted.${NC}"
    exit 1
}

# Log a message to the console
log() {
    local message="$1"
    local level="${2:-INFO}"
    local color="${NC}"
    
    case $level in
        INFO) color="${BLUE}" ;;
        SUCCESS) color="${GREEN}" ;;
        WARNING) color="${YELLOW}" ;;
        ERROR) color="${RED}" ;;
        DEBUG) color="${GRAY}" ;;
    esac
    
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] ${message}${NC}"
}

# Get current timestamp
get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Print formatted log messages
log_info() {
    log "$1" "INFO"
}

log_success() {
    log "$1" "SUCCESS"
}

log_warning() {
    log "$1" "WARNING"
}

log_error() {
    log "$1" "ERROR"
}

log_step() {
    echo -e "\n${GRAY}[$(get_timestamp)]${NC} ${BLUE}[STEP]${NC}    ${BOLD}$1${NC}"
}

log_debug() {
    if [[ "${DEBUG}" == "true" ]]; then
        log "$1" "DEBUG"
    fi
}

log_section() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}\n"
}

# Display title and logo
display_header() {
    local title="$1"
    
    echo ""
    echo -e "${CYAN}"
    echo ' ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗'
    echo ' ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝'
    echo ' ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  '
    echo ' ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  '
    echo ' ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗'
    echo ' ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝'
    echo -e "${NC}"
    echo -e "${BOLD}Second-Me Setup Script v${VERSION}${NC}"
    echo -e "${GRAY}$(date)${NC}\n"
    
    if [ -n "$title" ]; then
        echo -e "${CYAN}====== $title ======${NC}"
        echo ""
    fi
}

# Display stage start
display_stage() {
    local stage_num=$1
    local stage_name=$2
    CURRENT_STAGE=$stage_num
    STAGE_NAME=$stage_name
    
    echo ""
    echo -e "${CYAN}====== Stage $stage_num/$TOTAL_STAGES: $stage_name ======${NC}"
    echo ""
}

# Get conda environment name from .env file
get_conda_env_name() {
    if [[ -f ".env" ]]; then
        local env_name=$(grep '^CONDA_DEFAULT_ENV=' .env | cut -d '=' -f2)
        if [[ -n "$env_name" ]]; then
            echo "$env_name"
            return 0
        fi
    fi
    echo "second-me"  
    return 0
}

# Setup and configure package managers (npm)
setup_npm() {
    log_step "Setting up npm package manager"
    
    # Check if npm is already installed
    log_info "Checking npm installation..."
    if ! command -v npm &>/dev/null; then
        log_warning "npm not found - installing Node.js and npm"
        if ! brew install node; then
            log_error "Failed to install Node.js and npm"
            return 1
        fi
        
        # Verify npm was installed successfully
        if ! command -v npm &>/dev/null; then
            log_error "npm installation failed - command not found after installation"
            return 1
        fi
        log_success "Successfully installed Node.js and npm"
    else
        log_success "npm is already installed"
    fi
    
    # Configure npm settings
    log_info "Configuring npm settings..."
    
    # Set npm registry
    log_info "Setting npm registry to https://registry.npmjs.org/"
    npm config set registry https://registry.npmjs.org/
    
    # Set npm cache directory
    log_info "Setting npm cache directory to $HOME/.npm"
    npm config set cache "$HOME/.npm"
    
    # Verify npm configuration
    if npm config list &>/dev/null; then
        log_success "npm is properly configured"
    else
        log_error "npm configuration failed"
        return 1
    fi
    
    log_success "npm setup completed"
    return 0
}

# Check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# Install Homebrew packages
install_brew_packages() {
    log_step "Checking additional Homebrew packages"
    
    if ! check_command brew; then
        log_error "Homebrew is not installed."
        log_error "Please run the setup script from the beginning."
        return 1
    fi
    
    # Required brew packages
    local required_packages=(
        "sqlite"    # Database
        "make"      # Build tool
    )
    
    # Check and install each package
    for package in "${required_packages[@]}"; do
        if ! brew list $package &>/dev/null; then
            log_warning "Installing $package..."
            if ! brew install $package; then
                log_error "Failed to install $package"
                return 1
            fi
            
            # Verify installation
            if ! brew list $package &>/dev/null; then
                log_error "Failed to verify $package installation"
                return 1
            fi
        fi
    done
    
    log_success "All Homebrew packages are installed"
    return 0
}

# Install conda using Homebrew
install_conda() {
    log_info "Installing Miniconda using Homebrew..."
    
    brew install --cask miniconda
    local brew_result=$?
    
    if [ $brew_result -ne 0 ]; then
        log_error "Failed to install Miniconda (exit code: $brew_result)"
        return 1
    fi
    
    # Get Miniconda installation path from Homebrew cask info
    local conda_info
    conda_info=$(brew info --cask miniconda)
    if [ $? -ne 0 ]; then
        log_error "Failed to get Miniconda installation info"
        return 1
    fi
    
    # Extract conda binary path from brew info output
    local conda_binary
    conda_binary=$(echo "$conda_info" | grep "condabin/conda" | cut -d' ' -f1)
    if [ -z "$conda_binary" ]; then
        log_error "Could not find conda binary path in brew info output"
        return 1
    fi
    
    # Get the base directory (two levels up from condabin)
    local conda_root
    conda_root=$(dirname "$(dirname "$conda_binary")")
    if [ ! -d "$conda_root" ]; then
        log_error "Conda base directory not found at $conda_root"
        return 1
    fi
    
    log_info "Using Homebrew Miniconda installation path: $conda_root"
    
    # Initialize conda for bash and zsh
    log_info "Initializing conda for shell integration..."
    
    # First source conda.sh to make conda command available
    if ! find_and_source_conda_sh "$conda_root"; then
        log_error "Could not find conda.sh after installation at $conda_root"
        return 1
    fi
    
    # Now run conda init for both shells
    local shells=("zsh")
    for shell in "${shells[@]}"; do
        log_info "Running conda init for $shell..."
        if ! conda init "$shell"; then
            log_warning "Failed to initialize conda for $shell shell"
        else
            log_success "Initialized conda for $shell shell"
        fi
    done
    
    # Verify shell configurations
    for shell in "${shells[@]}"; do
        local rc_file="$HOME/.${shell}rc"
        if [ -f "$rc_file" ] && grep -q "conda initialize" "$rc_file"; then
            log_success "Verified conda initialization in $rc_file"
        else
            log_warning "Could not verify conda initialization in $rc_file"
        fi
    done
    
    # Add helpful message about shell restart
    log_info "Conda has been initialized for zsh shell"
    log_info "To use conda in a new shell, either:"
    log_info "1. Start a new shell session, or"
    log_info "2. Run: source ~/.zshrc"
    
    log_success "Conda installed and initialized successfully"
    return 0
}

# Activate Python environment
activate_python_env() {
    log_section "PYTHON ENVIRONMENT ACTIVATION"
    
    # If custom conda mode is enabled, skip to dependency installation
    if is_custom_conda_mode; then
        log_info "Using custom conda environment"
        goto_dependency_installation
        return $?
    fi
    
    # 1. Check conda installation
    log_step "Checking conda installation"
    if ! try_source_conda_sh_all; then
        log_error "Conda not found. It should have been installed during pre-installation checks."
        return 1
    fi
    
    # 2. Get environment name
    local env_name
    if ! env_name=$(get_conda_env_name); then
        log_error "Could not get conda environment name"
        return 1
    fi
    
    # 3. Create or update environment
    log_step "Activating conda environment: $env_name"
    
    if conda env list | grep -q "^${env_name} "; then
        log_info "Environment exists, updating..."
        if ! conda env update -f environment.yml -n "$env_name"; then
            log_error "Failed to update conda environment $env_name"
            return 1
        fi
    else
        log_info "Creating new environment... $env_name"
        if ! conda env create -f environment.yml -n "$env_name"; then
            log_error "Failed to create conda environment $env_name"
            return 1
        fi
    fi
    
    # attempt to activate conda environment
    log_info "Attempting to activate conda environment: $env_name"
    if ! conda activate "$env_name" 2>/dev/null; then
        log_info "conda activate failed, trying alternative method..."
        # if conda activate fails, try using source activate
        if command -v activate &>/dev/null && ! source activate "$env_name"; then
            log_error "Failed to activate conda environment using both methods"
            return 1
        fi
    fi
    
    log_success "Successfully activated conda environment: $env_name"
    goto_dependency_installation
    return $?
}

# Helper function to install dependencies
goto_dependency_installation() {
    # Install Python packages using Poetry
    log_step "Installing Python packages using Poetry"
    
    # Check if pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        log_error "Missing pyproject.toml file"
        return 1
    fi
    
    # Update lockfile and install dependencies
    log_info "Updating Poetry lockfile..."
    if ! poetry lock --no-cache; then
        log_error "Failed to update Poetry lockfile"
        return 1
    fi
    
    # Install dependencies
    log_info "Using Poetry to install dependencies..."
    if ! poetry install --no-root --no-interaction; then
        log_error "Failed to install dependencies using Poetry"
        return 1
    fi
    
    # Verify key packages are installed
    log_info "Verifying key packages..."
    local required_packages=("flask" "chromadb" "langchain")
    for pkg in "${required_packages[@]}"; do
        if ! python -c "import $pkg" 2>/dev/null; then
            log_error "Package '$pkg' is not installed correctly"
            return 1
        else
            log_info "Package '$pkg' is installed correctly"
        fi
    done

    # Check and ensure correct version of graphrag is installed
    log_step "Checking graphrag version"
    GRAPHRAG_VERSION=$(pip show graphrag 2>/dev/null | grep "Version:" | cut -d " " -f2)
    GRAPHRAG_TARGET="1.2.1.dev27"
    GRAPHRAG_LOCAL_PATH="dependencies/graphrag-${GRAPHRAG_TARGET}.tar.gz"

    if [ "$GRAPHRAG_VERSION" != "$GRAPHRAG_TARGET" ]; then
        log_info "Installing correct version of graphrag..."
        if [ -f "$GRAPHRAG_LOCAL_PATH" ]; then
            log_info "Installing graphrag from local file..."
            if ! pip install --force-reinstall "$GRAPHRAG_LOCAL_PATH"; then
                log_error "Failed to install graphrag from local file"
                return 1
            fi
        else
            log_error "Local graphrag package not found at: $GRAPHRAG_LOCAL_PATH"
            log_error "Please ensure the graphrag package exists in the dependencies directory"
            return 1
        fi
        log_success "Graphrag installed successfully"
    else
        log_success "Graphrag version is correct, skipping installation"
    fi

    log_success "Python environment setup completed"
    return 0
}

# Build llama.cpp
build_llama() {
    log_section "BUILDING LLAMA.CPP"
    
    LLAMA_LOCAL_ZIP="dependencies/llama.cpp.zip"
    
    # Check if llama.cpp directory exists
    if [ ! -d "llama.cpp" ]; then
        log_info "Setting up llama.cpp..."
        
        if [ -f "$LLAMA_LOCAL_ZIP" ]; then
            log_info "Using local llama.cpp archive..."
            if ! unzip -q "$LLAMA_LOCAL_ZIP"; then
                log_error "Failed to extract local llama.cpp archive"
                return 1
            fi
        else
            log_error "Local llama.cpp archive not found at: $LLAMA_LOCAL_ZIP"
            log_error "Please ensure the llama.cpp.zip file exists in the dependencies directory"
            return 1
        fi
    else
        log_info "Found existing llama.cpp directory"
    fi
    
    # Check if llama.cpp has been successfully compiled
    if [ -f "llama.cpp/build/bin/llama-server" ]; then
        log_info "Found existing llama-server build"
        # Check if executable file can be run and get version info
        if version_output=$(./llama.cpp/build/bin/llama-server --version 2>&1) && [[ $version_output == version:* ]]; then
            log_success "Existing llama-server build is working properly (${version_output}), skipping compilation"
            return 0
        else
            log_warning "Existing build seems broken or incompatible, will recompile..."
        fi
    fi
    
    # Enter llama.cpp directory and build
    cd llama.cpp
    
    # Clean previous build
    if [ -d "build" ]; then
        log_info "Cleaning previous build..."
        rm -rf build
    fi
    
    # Create and enter build directory
    log_info "Creating build directory..."
    mkdir -p build && cd build
    
    # Configure CMake
    log_info "Configuring CMake..."
    if ! cmake ..; then
        log_error "CMake configuration failed"
        cd ../..
        return 1
    fi
    
    # Build project
    log_info "Building project..."
    if ! cmake --build . --config Release; then
        log_error "Build failed"
        cd ../..
        return 1
    fi
    
    # Check build result
    if [ ! -f "bin/llama-server" ]; then
        log_error "Build failed: llama-server executable not found"
        log_error "Expected at: bin/llama-server"
        cd ../..
        return 1
    fi
    
    log_success "Found llama-server at: bin/llama-server"
    cd ../..
    log_section "LLAMA.CPP BUILD COMPLETE"
}

# Set up frontend environment
build_frontend() {
    log_section "SETTING UP FRONTEND"
    
    FRONTEND_DIR="lpm_frontend"
    
    # Enter frontend directory
    cd "$FRONTEND_DIR" || {
        log_error "Failed to enter frontend directory: $FRONTEND_DIR"
        log_error "Please ensure the directory exists and you have permission to access it."
        return 1
    }
    
    # Check if dependencies have been installed
    if [ -d "node_modules" ]; then
        log_info "Found existing node_modules, checking for updates..."
        if [ -f "package-lock.json" ]; then
            log_info "Using existing package-lock.json..."
            # Run npm install even if package-lock.json exists to ensure dependencies are complete
            log_info "Running npm install to ensure dependencies are complete..."
            if ! npm install; then
                log_error "Failed to install frontend dependencies with existing package-lock.json"
                log_error "Try removing node_modules directory and package-lock.json, then run setup again"
                cd ..
                return 1
            fi
        else
            log_info "Installing dependencies..."
            if ! npm install; then
                log_error "Failed to install frontend dependencies"
                log_error "Check your npm configuration and network connection"
                log_error "You can try running 'npm install' manually in the $FRONTEND_DIR directory"
                cd ..
                return 1
            fi
        fi
    else
        log_info "Installing dependencies..."
        if ! npm install; then
            log_error "Failed to install frontend dependencies"
            log_error "Check your npm configuration and network connection"
            log_error "You can try running 'npm install' manually in the $FRONTEND_DIR directory"
            cd ..
            return 1
        fi
    fi
    
    # Verify that the installation was successful
    if [ ! -d "node_modules" ]; then
        log_error "node_modules directory not found after npm install"
        log_error "Frontend dependencies installation failed"
        cd ..
        return 1
    fi
    
    log_success "Frontend dependencies installed successfully"
    cd ..
    log_section "FRONTEND SETUP COMPLETE"
}

# Initialize Conda environment if necessary
init_conda_env_if_necessary() {
    log_section "SETTING UP SHELL INTEGRATION"
    
    local config_file="$HOME/.zshrc"
    log_info "Will update shell configuration in: $config_file"
    
    # Add Conda
    if command -v conda &>/dev/null; then
        log_info "Checking Conda initialization status"
        # Check if already initialized
        if ! grep -q "conda initialize" "$config_file" 2>/dev/null; then
            log_info "Conda not initialized, running conda init"
            conda init zsh
            log_success "Added Conda initialization to shell configuration"
            
            # Source the updated config
            log_info "Applying new shell configuration..."
            source "$config_file"
            log_success "Shell configuration applied"
        else
            log_info "Conda already initialized"
        fi
    fi
    
    log_success "Conda environment initialized"
    return 0
}

# Ensure shell config file exists and is properly backed up
ensure_shell_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        log_info "Shell configuration file does not exist, creating it: $config_file"
        touch "$config_file"
        echo "# Shell configuration file created by Second-Me setup script on $(date)" > "$config_file"
        
        # Add basic configuration
        echo "# Set basic environment variables" >> "$config_file"
        echo 'export PATH="$HOME/bin:$HOME/.local/bin:$PATH"' >> "$config_file"
        
        log_success "Created new shell configuration file: $config_file"
    else
        # Backup config file
        cp "$config_file" "${config_file}.bak.$(date +%Y%m%d%H%M%S)"
        log_info "Created backup of $config_file"
    fi
    
    # Verify file is writable
    if [[ ! -w "$config_file" ]]; then
        log_error "Shell configuration file is not writable: $config_file"
        return 1
    fi
    
    return 0
}

# Show help information
show_help() {
    echo -e "${BOLD}Second-Me Setup Script v${VERSION}${NC}"
    echo -e "Usage: $0 [options] [command]"
    echo
    echo -e "Commands:"
    echo -e "  python\t\tSetup Python environment only"
    echo -e "  llama\t\t\tBuild llama.cpp only"
    echo -e "  frontend\t\tSetup frontend project only"
    echo -e "  (no command)\t\tPerform full installation"
    echo
    echo -e "Options:"
    echo -e "  --help\t\tShow this help information"
    echo -e "  --require-confirmation\tRequire confirmation when warnings are present"
    echo
    echo -e "Examples:"
    echo -e "  $0 \t\t\tPerform full installation"
    echo -e "  $0 python\t\tSetup Python environment only"
    echo -e "  $0 --require-confirmation\tRequire confirmation when warnings are present"
    echo
    echo -e "For a complete list of all available commands, run:"
    echo -e "  make help"
}

# Check system requirements
check_system_requirements() {
    log_section "CHECKING SYSTEM REQUIREMENTS"
    
    # Detect current shell
    local current_shell=$(basename "$SHELL")
    log_info "Detected shell: $current_shell"
    
    # Only support zsh
    if [[ "$current_shell" != "zsh" ]]; then
        log_error "Only zsh shell is supported"
        return 1
    fi

    # Check if running on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This script only supports macOS"
        return 1
    fi
    
    local macos_version=$(sw_vers -productVersion)
    log_info "Detected macOS version: $macos_version"
    
    local major_version=$(echo "$macos_version" | cut -d. -f1)
    if [[ "$major_version" -lt 14 ]]; then
        log_error "This script requires macOS 14 (Sonoma) or later. Your version: $macos_version"
        return 1
    fi

    # Check shell config file
    local config_file="$HOME/.zshrc"
    if ! ensure_shell_config "$config_file"; then
        log_error "Failed to setup shell configuration file"
        return 1
    fi
    
    # Rest of the system checks...
    
    # Detect system architecture
    local system_arch=$(uname -m)
    log_info "Detected system architecture: $system_arch"
    
    # Check installed Homebrew architecture
    if command -v brew &>/dev/null; then
        local brew_path=$(command -v brew)
        local brew_dir=$(dirname "$(dirname "$brew_path")")
        
        if [[ "$system_arch" == "arm64" && "$brew_dir" != "/opt/homebrew" ]]; then
            log_warning "Detected M1/M2 chip (arm64), but Homebrew installed in $brew_dir instead of /opt/homebrew"
            log_warning "This may indicate Homebrew was installed for Intel chips, which may cause performance issues"
            log_warning "It is recommended to uninstall current Homebrew and reinstall the arm64 version"
            # Do not force exit, just warn
        elif [[ "$system_arch" != "arm64" && "$brew_dir" == "/opt/homebrew" ]]; then
            log_warning "Detected Intel chip, but Homebrew installed in /opt/homebrew instead of /usr/local"
            log_warning "This may indicate Homebrew was installed for M1/M2 chips, which may cause compatibility issues"
            # Do not force exit, just warn
        else
            log_success "Homebrew architecture matches system architecture"
        fi
    fi
    
    # Check installed Conda architecture
    if command -v conda &>/dev/null; then
        local conda_info=$(conda info --json 2>/dev/null)
        if [[ $? -eq 0 ]]; then
            local conda_platform=$(echo "$conda_info" | grep -o '"platform": "[^"]*"' | cut -d'"' -f4)
            
            if [[ "$system_arch" == "arm64" && "$conda_platform" != *"arm64"* && "$conda_platform" != *"aarch64"* ]]; then
                log_error "Detected M1/M2 chip (arm64), but Conda seems to be installed for Intel chips (platform: $conda_platform)"
                log_error "This will cause performance issues or compatibility problems"
                log_error "Please uninstall current Conda and install Miniforge (a Conda distribution optimized for Apple Silicon)"
                return 1
            elif [[ "$system_arch" != "arm64" && ("$conda_platform" == *"arm64"* || "$conda_platform" == *"aarch64"*) ]]; then
                log_error "Detected Intel chip, but Conda seems to be installed for M1/M2 chips (platform: $conda_platform)"
                log_error "This will cause compatibility issues"
                log_error "Please uninstall current Conda and install the correct version for your architecture"
                return 1
            else
                log_success "Conda architecture matches system architecture"
            fi
        else
            log_warning "Failed to retrieve Conda information, skipping Conda architecture check"
        fi
    fi
    
    return 0
}

# Check required configuration files
check_config_files() {
    log_step "Checking necessary configuration files"
    
    # Check for .env file
    if [[ ! -f ".env" ]]; then
        log_error "Missing .env file"
        return 1
    fi
    
    # Check for environment.yml
    if [[ ! -f "environment.yml" ]]; then
        log_error "Missing environment.yml file"
        return 1
    fi
    
    log_success "All necessary configuration files are present"
    return 0
}

# Check directory permissions
check_directory_permissions() {
    log_step "Checking directory permissions"
    local errors=0
    local directories=("." "./scripts" "./run" "./logs")
    
    for dir in "${directories[@]}"; do
        if [[ ! -w "$dir" ]]; then
            log_error "Directory without write permission: $dir"
            errors=$((errors + 1))
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log_success "Directory permissions check passed"
        return 0
    else
        return 1
    fi
}

# Check for potential conflicts
check_potential_conflicts() {
    log_step "Checking for potential conflicts"

    # System requirements check
    if ! check_system_requirements; then
        log_error "System requirements check failed"
        exit 1
    fi
    
    # Check custom conda configuration if enabled
    if is_custom_conda_mode; then
        log_info "Custom conda mode is enabled, verifying environment..."
        if ! verify_conda_env; then
            log_error "Custom conda environment verification failed"
            exit 1
        fi
        log_success "Custom conda environment verification passed"
    fi
    
    # Configuration files check
    if ! check_config_files; then
        log_error "Configuration files check failed"
        exit 1
    fi
    
    # Directory permissions check
    if ! check_directory_permissions; then
        log_error "Directory permissions check failed"
        exit 1
    fi
    
    # Check Homebrew installation
    if command -v brew &>/dev/null; then
        log_info "Homebrew is installed"
    else
        log_warning "Homebrew is not installed, attempting to install it automatically..."
        
        # Only use local copy of the Homebrew install script
        local homebrew_script="${SCRIPT_DIR}/../dependencies/homebrew_install.sh"
        local brew_installed=false
        
        if [[ -f "$homebrew_script" ]]; then
            log_info "Using local Homebrew install script"
            if /bin/bash "$homebrew_script"; then
                
                # Add Homebrew to PATH for the current session
                if ! add_homebrew_to_path; then
                    log_error "Homebrew installed but couldn't be added to PATH"
                    return 1
                fi
                
                # Verify Homebrew is actually installed and working
                if command -v brew &>/dev/null; then
                    log_success "Homebrew installed successfully"
                else
                    log_error "Homebrew installation failed: brew command not found in PATH"
                    return 1
                fi
            fi
        else
            log_error "Local Homebrew install script not found at: $homebrew_script"
            log_error "Please ensure the Homebrew install script exists in the dependencies folder."
            return 1
        fi
    fi
    
    # Check Conda installation
    if command -v conda &>/dev/null; then
        log_info "Conda is installed"
    else
        log_warning "Conda is not installed, attempting to install it automatically..."
        # Check if Homebrew is available now
        if command -v brew &>/dev/null; then
            if ! install_conda; then
                log_error "Failed to install Conda automatically"
                return 1
            fi
        else
            log_error "Cannot install Conda: Homebrew is required but not available"
            return 1
        fi
    fi
    
    return 0
}

# Check and install cmake if not present
check_and_install_cmake() {
    log_step "Checking for cmake installation"
    
    if ! command -v cmake &>/dev/null; then
        log_warning "cmake is not installed, attempting to install it automatically..."
        if command -v brew &>/dev/null; then
            log_info "Installing cmake using Homebrew..."
            if ! brew install cmake; then
                log_error "Failed to install cmake using Homebrew"
                return 1
            fi
            log_success "cmake installed successfully"
        else
            log_error "Cannot install cmake: Homebrew is required but not available"
            return 1
        fi
    else
        log_info "cmake is installed"
    fi
    
    return 0
}

# Parse command line arguments
parse_args() {
    REQUIRE_CONFIRMATION=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --require-confirmation)
                REQUIRE_CONFIRMATION=true
                shift
                ;;
            python|llama|frontend)
                COMPONENT="$1"
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Add Homebrew to PATH for the current session
add_homebrew_to_path() {
    log_info "Adding Homebrew to PATH..."
    
    local brew_path="/opt/homebrew/bin"
    local config_file="$HOME/.zshrc"
    
    if [[ -f "$brew_path/brew" ]]; then
        log_info "Found Homebrew installation at: $brew_path"
        
        # Set up Homebrew environment for current session
        eval "$(/opt/homebrew/bin/brew shellenv)"
        
        # Add Homebrew initialization to .zshrc if not already present
        if ! grep -q "HOMEBREW_PREFIX" "$config_file" 2>/dev/null; then
            log_info "Adding Homebrew initialization to $config_file"
            echo "" >> "$config_file"
            echo "# Set up Homebrew environment" >> "$config_file"
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$config_file"
            
            # Source the updated config
            log_info "Applying new shell configuration..."
            source "$config_file"
            log_success "Shell configuration applied"
        else
            log_info "Homebrew initialization already present in shell configuration"
        fi
        
        log_success "Homebrew added to PATH successfully"
        return 0
    fi
    
    log_error "Could not find Homebrew installation at $brew_path"
    return 1
}

# Main function
main() {
    # Display welcome message
    display_header "Second-Me Complete Installation"
    
    # Parse command line arguments
    parse_args "$@"
    
    # All pre-installation checks
    log_section "Running pre-installation checks"
    
    # 1. Basic tools check (most fundamental)
    # install homebrew, conda if necessary
    if ! check_potential_conflicts; then
        log_error "Basic tools check failed"
        exit 1
    fi
    
    # Start installation process
    log_section "Starting installation"
    
    # 1. Setup Conda environment
    if ! activate_python_env; then
        exit 1
    fi
    
    # 2. Setup npm
    if ! setup_npm; then
        log_error "npm setup failed"
        exit 1
    fi
    
    # 2. Check and install cmake
    if ! check_and_install_cmake; then
        log_error "cmake check and installation failed"
        exit 1
    fi
    
    # 3. Build llama.cpp
    if ! build_llama; then
        exit 1
    fi
    
    # 4. Build frontend
    if ! build_frontend; then
        exit 1
    fi
    
    # # 5. Initialize Conda environment
    # if ! init_conda_env_if_necessary; then
    #     exit 1
    # fi

    # Source the shell configuration to ensure all changes take effect
    local config_file="$HOME/.zshrc"
    if [[ -f "$config_file" ]]; then
        log_info "Applying final shell configuration..."
        source "$config_file"
        log_success "Shell configuration applied"
    fi

    log_success "Installation complete!"
    return 0
}

# Start execution
main "$@"
