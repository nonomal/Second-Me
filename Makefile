.PHONY: install test format lint all setup start stop restart restart-backend restart-force help docker-build docker-up docker-down docker-build-backend docker-build-frontend docker-restart-backend docker-restart-frontend docker-restart-all

# Detect operating system and set environment
ifeq ($(OS),Windows_NT)
    # Set Windows variables
    WINDOWS := 1
    # Set UTF-8 code page for Windows to display Unicode characters
    SET_UTF8 := $(shell chcp 65001 >nul 2>&1 || echo)
    # No need to check for Apple Silicon on Windows
    APPLE_SILICON := 0
    # Define empty color codes for Windows to avoid display issues
    COLOR_CYAN := 
    COLOR_RESET := 
    COLOR_BOLD := 
    COLOR_GRAY := 
    COLOR_GREEN := 
    COLOR_RED := 
else
    WINDOWS := 0
    # Detect Apple Silicon on non-Windows systems
    ifeq ($(shell uname -s),Darwin)
      ifeq ($(shell uname -m),arm64)
        APPLE_SILICON := 1
        # Set environment variables for Apple Silicon
        export DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.apple
        export PLATFORM=apple
      else
        APPLE_SILICON := 0
      endif
    else
      APPLE_SILICON := 0
    endif
    # Define ANSI color codes for Unix systems
    COLOR_CYAN := \033[0;36m
    COLOR_RESET := \033[0m
    COLOR_BOLD := \033[1m
    COLOR_GRAY := \033[0;90m
    COLOR_GREEN := \033[1;32m
    COLOR_RED := \033[1;31m
endif

# Show help message
help:
ifeq ($(WINDOWS),1)
	@echo.
	@echo  ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗
	@echo  ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝
	@echo  ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  
	@echo  ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  
	@echo  ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗
	@echo  ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝
	@echo.
	@echo SECOND-ME MAKEFILE COMMANDS
	@echo ------------------------------
	@echo.
	@echo LOCAL COMMANDS:
	@echo   make setup                 - Complete installation
	@echo   make start                 - Start all services
	@echo   make stop                  - Stop all services
	@echo   make restart               - Restart all services
	@echo   make restart-backend       - Restart only backend service
	@echo   make restart-force         - Force restart and reset data
	@echo   make status                - Show status of all services
	@echo.
	@echo DOCKER COMMANDS:
	@echo   make docker-build          - Build all Docker images
	@echo   make docker-up             - Start all Docker containers
	@echo   make docker-down           - Stop all Docker containers
	@echo   make docker-build-backend  - Build only backend Docker image
	@echo   make docker-build-frontend - Build only frontend Docker image
	@echo   make docker-restart-backend - Restart only backend container
	@echo   make docker-restart-frontend - Restart only frontend container
	@echo   make docker-restart-all    - Restart all Docker containers
	@echo.
	@echo All Available Commands:
	@echo   make help                  - Show this help message
	@echo   make install               - Install project dependencies
	@echo   make test                  - Run tests
	@echo   make format                - Format code
	@echo   make lint                  - Check code style
	@echo   make all                   - Run format, lint and test
else
	@echo "$(COLOR_CYAN)"
	@echo ' ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗'
	@echo ' ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝'
	@echo ' ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  '
	@echo ' ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  '
	@echo ' ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗'
	@echo ' ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝'
	@echo "$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)Second-Me Makefile Commands$(COLOR_RESET)"
	@echo "$(COLOR_GRAY)$$(date)$(COLOR_RESET)\n"
	@echo ""
	@echo "$(COLOR_GREEN)▶ LOCAL COMMANDS:$(COLOR_RESET)"
	@echo "  make setup                 - Complete installation"
	@echo "  make start                 - Start all services"
	@echo "  make stop                  - Stop all services"
	@echo "  make restart               - Restart all services"
	@echo "  make restart-backend       - Restart only backend service"
	@echo "  make restart-force         - Force restart and reset data"
	@echo "  make status                - Show status of all services"
	@echo ""
	@echo "$(COLOR_GREEN)▶ DOCKER COMMANDS:$(COLOR_RESET)"
	@echo "  make docker-build          - Build all Docker images"
	@echo "  make docker-up             - Start all Docker containers"
	@echo "  make docker-down           - Stop all Docker containers"
	@echo "  make docker-build-backend  - Build only backend Docker image"
	@echo "  make docker-build-frontend - Build only frontend Docker image"
	@echo "  make docker-restart-backend - Restart only backend container"
	@echo "  make docker-restart-frontend - Restart only frontend container"
	@echo "  make docker-restart-all    - Restart all Docker containers"
	@echo ""
	@echo "$(COLOR_BOLD)All Available Commands:$(COLOR_RESET)"
	@echo "  make help                  - Show this help message"
	@echo "  make install               - Install project dependencies"
	@echo "  make test                  - Run tests"
	@echo "  make format                - Format code"
	@echo "  make lint                  - Check code style"
	@echo "  make all                   - Run format, lint and test"
	@if [ "$(APPLE_SILICON)" = "1" ]; then \
		echo ""; \
		echo "$(COLOR_GREEN)▶ PLATFORM INFORMATION:$(COLOR_RESET)"; \
		echo "  Apple Silicon detected - Docker commands will use PLATFORM=apple"; \
	fi
endif

setup:
	./scripts/setup.sh

start:
	./scripts/start.sh

stop:
	./scripts/stop.sh

restart:
	./scripts/restart.sh

restart-backend:
	./scripts/restart-backend.sh

restart-force:
	./scripts/restart-force.sh

status:
	./scripts/status.sh

# Docker commands
# Set Docker environment variable for all Docker commands
docker-%: export IN_DOCKER_ENV=1

ifeq ($(OS),Windows_NT)
DOCKER_COMPOSE_CMD := docker compose
else
DOCKER_COMPOSE_CMD := $(shell if command -v docker-compose >/dev/null 2>&1; then echo "docker-compose"; else echo "docker compose"; fi)
endif

docker-build:
	$(DOCKER_COMPOSE_CMD) build

docker-up:
	$(DOCKER_COMPOSE_CMD) up -d

docker-down:
	$(DOCKER_COMPOSE_CMD) down

docker-build-backend:
	$(DOCKER_COMPOSE_CMD) build backend

docker-build-frontend:
	$(DOCKER_COMPOSE_CMD) build frontend

docker-restart-backend:
	$(DOCKER_COMPOSE_CMD) stop backend
	$(DOCKER_COMPOSE_CMD) rm -f backend
	$(DOCKER_COMPOSE_CMD) build backend || { echo "$(COLOR_RED)❌ Backend build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	$(DOCKER_COMPOSE_CMD) up -d backend

docker-restart-frontend:
	$(DOCKER_COMPOSE_CMD) stop frontend
	$(DOCKER_COMPOSE_CMD) rm -f frontend
	$(DOCKER_COMPOSE_CMD) build frontend || { echo "$(COLOR_RED)❌ Frontend build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	$(DOCKER_COMPOSE_CMD) up -d frontend

docker-restart-all:
	$(DOCKER_COMPOSE_CMD) stop
	$(DOCKER_COMPOSE_CMD) rm -f
	$(DOCKER_COMPOSE_CMD) build || { echo "$(COLOR_RED)❌ Build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	$(DOCKER_COMPOSE_CMD) up -d

install:
	poetry install

test:
	poetry run pytest tests

format:
	poetry run ruff format lpm_kernel/

lint:
	poetry run ruff check lpm_kernel/

all: format lint test