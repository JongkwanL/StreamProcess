.PHONY: help install dev-install proto build run stop clean test lint format benchmark

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := streamprocess

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)StreamProcess - Real-time Multimodal Processing Pipeline$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

dev-install: install ## Install development dependencies
	$(PIP) install -r requirements-dev.txt 2>/dev/null || true
	pre-commit install 2>/dev/null || true
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

proto: ## Compile protobuf files
	$(PYTHON) scripts/compile_protos.py
	@echo "$(GREEN)✓ Protobuf files compiled$(NC)"

build: proto ## Build Docker images
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Docker images built$(NC)"

run: ## Start all services with docker-compose
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  - gRPC Server: localhost:50051"
	@echo "  - REST API: http://localhost:8000"
	@echo "  - Redis: localhost:6379"
	@echo "  - MinIO: http://localhost:9001 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9091"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"

run-dev: ## Start services in development mode
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up

stop: ## Stop all services
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

clean: stop ## Clean up containers, volumes, and generated files
	$(DOCKER_COMPOSE) down -v
	rm -rf src/generated/*.py
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf logs/*.log
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-unit: ## Run unit tests only
	pytest tests/unit -v

test-integration: ## Run integration tests
	pytest tests/integration -v

lint: ## Run linting checks
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	black src/ tests/ --line-length=100
	@echo "$(GREEN)✓ Code formatted$(NC)"

benchmark: ## Run performance benchmarks
	$(PYTHON) benchmarks/run_benchmarks.py
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

logs-grpc: ## Show gRPC server logs
	$(DOCKER_COMPOSE) logs -f grpc-server

logs-workers: ## Show worker logs
	$(DOCKER_COMPOSE) logs -f stt-worker ocr-worker

monitor: ## Open monitoring dashboards
	@echo "$(YELLOW)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9091"

download-models: ## Download ML models
	$(PYTHON) scripts/download_models.py
	@echo "$(GREEN)✓ Models downloaded$(NC)"

scale-stt: ## Scale STT workers (usage: make scale-stt N=4)
	$(DOCKER_COMPOSE) up -d --scale stt-worker=$(N)
	@echo "$(GREEN)✓ Scaled STT workers to $(N)$(NC)"

scale-ocr: ## Scale OCR workers (usage: make scale-ocr N=4)
	$(DOCKER_COMPOSE) up -d --scale ocr-worker=$(N)
	@echo "$(GREEN)✓ Scaled OCR workers to $(N)$(NC)"

health: ## Check service health
	@echo "$(YELLOW)Checking service health...$(NC)"
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@redis-cli ping 2>/dev/null && echo "$(GREEN)✓ Redis is healthy$(NC)" || echo "$(RED)✗ Redis is not responding$(NC)"

init: install proto download-models ## Initialize project (install deps, compile protos, download models)
	@echo "$(GREEN)✓ Project initialized successfully!$(NC)"

dev: dev-install proto ## Setup development environment
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

.DEFAULT_GOAL := help