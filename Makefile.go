# StreamProcess Go Services Makefile
# High-performance Go implementation with Podman

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Go build configuration
GOOS ?= linux
GOARCH ?= amd64
CGO_ENABLED ?= 0
LDFLAGS := -w -s -extldflags '-static'
BUILD_DIR := bin

# Docker/Podman configuration
REGISTRY ?= localhost
TAG ?= latest
PODMAN_CMD := podman

# Application configuration
APP_NAME := streamprocess
NAMESPACE := streamprocess

## Build targets

.PHONY: deps
deps: ## Install Go dependencies
	@echo "Installing Go dependencies..."
	go mod download
	go mod verify

.PHONY: protobuf
protobuf: ## Generate Go protobuf files
	@echo "Generating Go protobuf files..."
	@mkdir -p protos/go
	protoc --go_out=protos/go --go_opt=paths=source_relative \
		--go-grpc_out=protos/go --go-grpc_opt=paths=source_relative \
		--grpc-gateway_out=protos/go --grpc-gateway_opt=paths=source_relative \
		protos/stream_process.proto

.PHONY: build
build: build-grpc-server build-stt-worker build-ocr-worker build-autoscaler ## Build all Go binaries

.PHONY: build-grpc-server
build-grpc-server: ## Build gRPC server
	@echo "Building gRPC server..."
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="$(LDFLAGS)" -o $(BUILD_DIR)/grpc-server ./cmd/grpc-server

.PHONY: build-stt-worker
build-stt-worker: ## Build STT worker
	@echo "Building STT worker..."
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="$(LDFLAGS)" -o $(BUILD_DIR)/stt-worker ./cmd/stt-worker

.PHONY: build-ocr-worker
build-ocr-worker: ## Build OCR worker
	@echo "Building OCR worker..."
	CGO_ENABLED=1 GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="$(LDFLAGS)" -o $(BUILD_DIR)/ocr-worker ./cmd/ocr-worker

.PHONY: build-autoscaler
build-autoscaler: ## Build autoscaler controller
	@echo "Building autoscaler..."
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="$(LDFLAGS)" -o $(BUILD_DIR)/autoscaler ./cmd/autoscaler

## Container targets

.PHONY: podman-build
podman-build: podman-build-grpc podman-build-stt podman-build-ocr podman-build-autoscaler ## Build all containers

.PHONY: podman-build-grpc
podman-build-grpc: ## Build gRPC server container
	@echo "Building gRPC server container..."
	$(PODMAN_CMD) build -f Containerfile.go --target grpc-server \
		-t $(REGISTRY)/$(APP_NAME):grpc-server-$(TAG) .

.PHONY: podman-build-stt
podman-build-stt: ## Build STT worker container
	@echo "Building STT worker container..."
	$(PODMAN_CMD) build -f Containerfile.go --target stt-worker \
		-t $(REGISTRY)/$(APP_NAME):stt-worker-$(TAG) .

.PHONY: podman-build-ocr
podman-build-ocr: ## Build OCR worker container
	@echo "Building OCR worker container..."
	$(PODMAN_CMD) build -f Containerfile.go --target ocr-worker \
		-t $(REGISTRY)/$(APP_NAME):ocr-worker-$(TAG) .

.PHONY: podman-build-autoscaler
podman-build-autoscaler: ## Build autoscaler container
	@echo "Building autoscaler container..."
	$(PODMAN_CMD) build -f Containerfile.go --target autoscaler \
		-t $(REGISTRY)/$(APP_NAME):autoscaler-$(TAG) .

.PHONY: podman-up
podman-up: ## Start all services with podman-compose
	@echo "Starting StreamProcess with Podman (Go services)..."
	podman-compose -f podman-compose-go.yml up -d

.PHONY: podman-down
podman-down: ## Stop all services
	@echo "Stopping StreamProcess services..."
	podman-compose -f podman-compose-go.yml down

.PHONY: podman-logs
podman-logs: ## Show logs from all services
	podman-compose -f podman-compose-go.yml logs -f

.PHONY: podman-clean
podman-clean: ## Clean up containers and images
	@echo "Cleaning up Podman containers and images..."
	podman-compose -f podman-compose-go.yml down --volumes
	podman image prune -f
	podman container prune -f

## Security and validation

.PHONY: security-scan
security-scan: ## Run security scans on containers
	@echo "Running security scans..."
	for image in grpc-server stt-worker ocr-worker autoscaler; do \
		echo "Scanning $(REGISTRY)/$(APP_NAME):$$image-$(TAG)"; \
		podman run --rm \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v /tmp:/tmp \
			aquasec/trivy image $(REGISTRY)/$(APP_NAME):$$image-$(TAG); \
	done

.PHONY: rootless-check
rootless-check: ## Verify rootless operation
	@echo "Checking rootless operation..."
	@if podman info --format '{{.Host.Security.Rootless}}' | grep -q true; then \
		echo "✓ Running in rootless mode"; \
	else \
		echo "✗ Not running in rootless mode"; \
		exit 1; \
	fi

.PHONY: validate-containers
validate-containers: ## Validate container security settings
	@echo "Validating container security settings..."
	@for service in grpc-server stt-worker ocr-worker autoscaler; do \
		echo "Checking $$service security..."; \
		podman inspect $(REGISTRY)/$(APP_NAME):$$service-$(TAG) \
			--format '{{.Config.User}}' | grep -q appuser || \
			(echo "✗ $$service not running as non-root user" && exit 1); \
		echo "✓ $$service running as non-root user"; \
	done

## Testing

.PHONY: test
test: test-unit test-integration ## Run all tests

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "Running unit tests..."
	go test -v -race -coverprofile=coverage.out ./...

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "Running integration tests..."
	go test -v -tags=integration ./tests/integration/...

.PHONY: test-load
test-load: ## Run load tests
	@echo "Running load tests..."
	go run ./benchmarks/load_test.go

.PHONY: benchmark
benchmark: ## Run Go benchmarks
	@echo "Running Go benchmarks..."
	go test -bench=. -benchmem ./...

## Development

.PHONY: dev-setup
dev-setup: deps protobuf ## Setup development environment
	@echo "Setting up development environment..."
	@mkdir -p $(BUILD_DIR)
	@mkdir -p logs
	@mkdir -p models

.PHONY: run-grpc
run-grpc: build-grpc-server ## Run gRPC server locally
	@echo "Starting gRPC server..."
	./$(BUILD_DIR)/grpc-server

.PHONY: run-stt-worker
run-stt-worker: build-stt-worker ## Run STT worker locally
	@echo "Starting STT worker..."
	./$(BUILD_DIR)/stt-worker

.PHONY: run-ocr-worker
run-ocr-worker: build-ocr-worker ## Run OCR worker locally
	@echo "Starting OCR worker..."
	./$(BUILD_DIR)/ocr-worker

.PHONY: run-autoscaler
run-autoscaler: build-autoscaler ## Run autoscaler locally
	@echo "Starting autoscaler..."
	./$(BUILD_DIR)/autoscaler

## Monitoring

.PHONY: metrics
metrics: ## Show metrics from all services
	@echo "Collecting metrics..."
	@echo "=== gRPC Server Metrics ==="
	@curl -s http://localhost:9092/metrics | head -20
	@echo "=== STT Worker Metrics ==="
	@curl -s http://localhost:9090/metrics | head -20
	@echo "=== OCR Worker Metrics ==="
	@curl -s http://localhost:9091/metrics | head -20
	@echo "=== Autoscaler Metrics ==="
	@curl -s http://localhost:9093/metrics | head -20

.PHONY: health-check
health-check: ## Check health of all services
	@echo "Checking service health..."
	@echo "gRPC Server: $$(curl -s -w '%{http_code}' http://localhost:9092/health)"
	@echo "STT Worker: $$(curl -s -w '%{http_code}' http://localhost:9090/health)"
	@echo "OCR Worker: $$(curl -s -w '%{http_code}' http://localhost:9091/health)"
	@echo "Autoscaler: $$(curl -s -w '%{http_code}' http://localhost:9093/health)"

## Kubernetes deployment

.PHONY: k8s-deploy
k8s-deploy: ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/redis.yaml
	kubectl apply -f k8s/workers-go.yaml
	kubectl apply -f k8s/grpc-server-go.yaml
	kubectl apply -f k8s/autoscaler-go.yaml

.PHONY: k8s-update
k8s-update: podman-build k8s-deploy ## Build and update Kubernetes deployment
	@echo "Updating Kubernetes deployment..."
	kubectl rollout restart deployment/grpc-server-go -n $(NAMESPACE)
	kubectl rollout restart deployment/stt-worker-go -n $(NAMESPACE)
	kubectl rollout restart deployment/ocr-worker-go -n $(NAMESPACE)
	kubectl rollout restart deployment/autoscaler-go -n $(NAMESPACE)

.PHONY: k8s-status
k8s-status: ## Check Kubernetes deployment status
	@echo "Checking Kubernetes status..."
	kubectl get pods -n $(NAMESPACE)
	kubectl get services -n $(NAMESPACE)
	kubectl get hpa -n $(NAMESPACE)

## Performance optimization

.PHONY: profile-cpu
profile-cpu: ## Run CPU profiling
	@echo "Running CPU profiling..."
	go test -cpuprofile=cpu.prof -bench=. ./pkg/...
	go tool pprof cpu.prof

.PHONY: profile-memory
profile-memory: ## Run memory profiling
	@echo "Running memory profiling..."
	go test -memprofile=mem.prof -bench=. ./pkg/...
	go tool pprof mem.prof

.PHONY: optimize
optimize: ## Run optimization checks
	@echo "Running optimization analysis..."
	go run -gcflags=-m ./cmd/grpc-server 2>&1 | grep "inlining\|escapes"

## Cleanup

.PHONY: clean
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f coverage.out
	rm -f *.prof
	rm -f logs/*.log

.PHONY: clean-all
clean-all: clean podman-clean ## Clean everything
	@echo "Cleaning everything..."
	go clean -cache
	go clean -modcache

## Documentation

.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	godoc -http=:8080

.PHONY: api-docs
api-docs: ## Generate API documentation
	@echo "Generating API documentation..."
	swagger generate spec -o api/swagger.yaml

## Help

.PHONY: help
help: ## Show this help message
	@echo "StreamProcess Go Services Makefile"
	@echo "Usage: make [target]"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: info
info: ## Show build information
	@echo "Build Information:"
	@echo "  GOOS: $(GOOS)"
	@echo "  GOARCH: $(GOARCH)"
	@echo "  CGO_ENABLED: $(CGO_ENABLED)"
	@echo "  BUILD_DIR: $(BUILD_DIR)"
	@echo "  REGISTRY: $(REGISTRY)"
	@echo "  TAG: $(TAG)"
	@echo "  PODMAN_CMD: $(PODMAN_CMD)"