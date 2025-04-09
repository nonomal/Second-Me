#!/bin/bash
# Installation recommendations configuration

# Get installation recommendation for a package
get_install_recommendation() {
    local package="$1"
    local system_id="$2"
    
    case "$package" in
        "python")
            get_python_recommendation "$system_id"
            ;;
        "npm")
            get_npm_recommendation "$system_id"
            ;;
        "node")
            get_node_recommendation "$system_id"
            ;;
        *)
            echo "No specific recommendation available for $package"
            ;;
    esac
}

# Python installation recommendations
get_python_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            echo "Recommended installation for macOS: 'brew install python3'"
            echo "Or download from: https://www.python.org/downloads/macos/"
            ;;
        "linux-debian")
            echo "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install python3 python3-pip'"
            ;;
        "linux-fedora")
            echo "Recommended installation for Fedora: 'sudo dnf install python3 python3-pip'"
            ;;
        "linux-redhat")
            echo "Recommended installation for CentOS/RHEL: 'sudo yum install python3 python3-pip'"
            ;;
        "linux-arch")
            echo "Recommended installation for Arch Linux: 'sudo pacman -S python python-pip'"
            ;;
        "linux-alpine")
            echo "Recommended installation for Alpine Linux: 'apk add python3 py3-pip'"
            ;;
        "linux-other")
            echo "Please install Python 3.12+ using your distribution's package manager"
            echo "Or download from: https://www.python.org/downloads/linux/"
            ;;
        "windows")
            echo "Recommended installation for Windows:"
            echo "1. Download from: https://www.python.org/downloads/windows/"
            echo "2. Or using winget: 'winget install Python.Python.3'"
            echo "3. Or using Chocolatey: 'choco install python'"
            ;;
        *)
            echo "Please download Python from: https://www.python.org/downloads/"
            ;;
    esac
}

# NPM installation recommendations
get_npm_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            echo "Recommended installation for macOS: 'brew install npm'"
            ;;
        "linux-debian")
            echo "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install npm'"
            ;;
        "linux-fedora")
            echo "Recommended installation for Fedora: 'sudo dnf install npm'"
            ;;
        "linux-redhat")
            echo "Recommended installation for CentOS/RHEL: 'sudo yum install npm'"
            ;;
        "linux-arch")
            echo "Recommended installation for Arch Linux: 'sudo pacman -S npm'"
            ;;
        "linux-alpine")
            echo "Recommended installation for Alpine Linux: 'apk add npm'"
            ;;
        "linux-other")
            echo "Please install npm using your distribution's package manager"
            ;;
        "windows")
            echo "Recommended installation for Windows:"
            echo "1. Install Node.js (includes npm): https://nodejs.org/en/download/"
            echo "2. Or using winget: 'winget install OpenJS.NodeJS'"
            echo "3. Or using Chocolatey: 'choco install nodejs'"
            ;;
        *)
            echo "Please install Node.js (includes npm): https://nodejs.org/en/download/"
            ;;
    esac
}

# Node.js installation recommendations
get_node_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            echo "Recommended installation for macOS: 'brew install node'"
            ;;
        "linux-debian")
            echo "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install nodejs'"
            ;;
        "linux-fedora")
            echo "Recommended installation for Fedora: 'sudo dnf install nodejs'"
            ;;
        "linux-redhat")
            echo "Recommended installation for CentOS/RHEL: 'sudo yum install nodejs'"
            ;;
        "linux-arch")
            echo "Recommended installation for Arch Linux: 'sudo pacman -S nodejs'"
            ;;
        "linux-alpine")
            echo "Recommended installation for Alpine Linux: 'apk add nodejs'"
            ;;
        "linux-other")
            echo "Please install Node.js using your distribution's package manager"
            ;;
        "windows")
            echo "Recommended installation for Windows:"
            echo "1. Download from: https://nodejs.org/en/download/"
            echo "2. Or using winget: 'winget install OpenJS.NodeJS'"
            echo "3. Or using Chocolatey: 'choco install nodejs'"
            ;;
        *)
            echo "Please download Node.js from: https://nodejs.org/en/download/"
            ;;
    esac
}
