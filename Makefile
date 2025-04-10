# Available Commands:
# Commands that don't require conda environment:
#   make help         - Show this help message
#   make setup        - Run complete installation script
#   make start        - Start all services
#   make stop         - Stop chat services
#   make restart      - Restart chat services
#   make restart-backend - Restart only backend service
#   make restart-force - Force restart and reset data
#   make status       - Show status of all services

.PHONY: install test format lint all setup start stop restart restart-backend restart-force help check-conda check-env docker-build docker-up docker-down docker-build-backend docker-build-frontend docker-restart-backend docker-restart-frontend docker-restart-all

# Detect Apple Silicon without printing
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

# Show help message
help:
	@echo "\033[0;36m"
	@echo ' ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗'
	@echo ' ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝'
	@echo ' ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  '
	@echo ' ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  '
	@echo ' ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗'
	@echo ' ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝'
	@echo "\033[0m"
	@echo "\033[1mSecond-Me Makefile Commands\033[0m"
	@echo "\033[0;90m$$(date)\033[0m\n"
	@echo ""
	@echo "\033[1;32m▶ MAIN COMMANDS:\033[0m"
	@echo "  make setup                 - Complete installation"
	@echo "  make start                 - Start all services"
	@echo "  make stop                  - Stop all services"
	@echo "  make restart               - Restart all services"
	@echo "  make restart-backend       - Restart only backend service"
	@echo "  make restart-force         - Force restart and reset data"
	@echo "  make status                - Show status of all services"
	@echo ""
	@echo "\033[1;32m▶ DOCKER COMMANDS:\033[0m"
	@echo "  make docker-build          - Build all Docker images"
	@echo "  make docker-up             - Start all Docker containers"
	@echo "  make docker-down           - Stop all Docker containers"
	@echo "  make docker-build-backend  - Build only backend Docker image"
	@echo "  make docker-build-frontend - Build only frontend Docker image"
	@echo "  make docker-restart-backend - Restart only backend container"
	@echo "  make docker-restart-frontend - Restart only frontend container"
	@echo "  make docker-restart-all    - Restart all Docker containers"
	@echo ""
	@echo "\033[1mAll Available Commands:\033[0m"
	@echo "  make help                  - Show this help message"
	@echo "  make check-env             - Check environment without installing"
	@echo ""
	@echo "\033[1mCommands that require conda environment:\033[0m"
	@echo "  make install               - Install project dependencies"
	@echo "  make test                  - Run tests"
	@echo "  make format                - Format code"
	@echo "  make lint                  - Check code style"
	@echo "  make all                   - Run format, lint and test"
	@if [ "$(APPLE_SILICON)" = "1" ]; then \
		echo ""; \
		echo "\033[1;32m▶ PLATFORM INFORMATION:\033[0m"; \
		echo "  Apple Silicon detected - Docker commands will use PLATFORM=apple"; \
	fi

# Check if in conda environment
check-conda:
	@if [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "Error: Please activate conda environment first!"; \
		exit 1; \
	fi

# Check environment without installing
check-env:
	zsh ./scripts/setup.sh --check-only

# Commands that don't require conda environment
setup:
	zsh ./scripts/setup.sh

start:
	zsh ./scripts/start.sh

stop:
	zsh ./scripts/stop.sh

restart:
	zsh ./scripts/restart.sh

restart-backend:
	zsh ./scripts/restart-backend.sh

restart-force:
	zsh ./scripts/restart-force.sh

status:
	zsh ./scripts/status.sh

# Docker commands
# Set Docker environment variable for all Docker commands
docker-%: export IN_DOCKER_ENV=1

# 检测是否安装了 docker compose 插件
DOCKER_COMPOSE_CMD := $(shell if command -v docker compose >/dev/null 2>&1; then echo "docker compose"; else echo "docker-compose"; fi)

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
	$(DOCKER_COMPOSE_CMD) build backend || { echo "\033[1;31m❌ Backend build failed! Aborting operation...\033[0m"; exit 1; }
	$(DOCKER_COMPOSE_CMD) up -d backend

docker-restart-frontend:
	$(DOCKER_COMPOSE_CMD) stop frontend
	$(DOCKER_COMPOSE_CMD) rm -f frontend
	$(DOCKER_COMPOSE_CMD) build frontend || { echo "\033[1;31m❌ Frontend build failed! Aborting operation...\033[0m"; exit 1; }
	$(DOCKER_COMPOSE_CMD) up -d frontend

docker-restart-all:
	$(DOCKER_COMPOSE_CMD) stop
	$(DOCKER_COMPOSE_CMD) rm -f
	$(DOCKER_COMPOSE_CMD) build || { echo "\033[1;31m❌ Build failed! Aborting operation...\033[0m"; exit 1; }
	$(DOCKER_COMPOSE_CMD) up -d

# Commands that require conda environment
install: check-conda
	poetry install

test: check-conda
	poetry run pytest tests

format: check-conda
	poetry run ruff format lpm_kernel/

lint: check-conda
	poetry run ruff check lpm_kernel/

all: check-conda format lint test