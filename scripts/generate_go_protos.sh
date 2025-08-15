#!/bin/bash
# Generate Go protobuf files from .proto definitions

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$PROJECT_ROOT/protos"
GO_OUT_DIR="$PROJECT_ROOT/pkg/protos"

echo "Generating Go protobuf files..."
echo "Project root: $PROJECT_ROOT"
echo "Proto dir: $PROTO_DIR"
echo "Go output dir: $GO_OUT_DIR"

# Create output directory
mkdir -p "$GO_OUT_DIR"

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "Error: protoc compiler not found. Please install Protocol Buffers compiler."
    echo "macOS: brew install protobuf"
    echo "Ubuntu: sudo apt-get install protobuf-compiler"
    exit 1
fi

# Check if protoc-gen-go is installed
if ! command -v protoc-gen-go &> /dev/null; then
    echo "Installing protoc-gen-go..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

# Check if protoc-gen-go-grpc is installed
if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "Installing protoc-gen-go-grpc..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

# Find all .proto files
PROTO_FILES=($(find "$PROTO_DIR" -name "*.proto"))

if [ ${#PROTO_FILES[@]} -eq 0 ]; then
    echo "No .proto files found in $PROTO_DIR"
    exit 1
fi

echo "Found ${#PROTO_FILES[@]} proto file(s)"

# Generate Go files
for proto_file in "${PROTO_FILES[@]}"; do
    echo "Generating Go code for $(basename "$proto_file")..."
    
    protoc \
        --proto_path="$PROTO_DIR" \
        --go_out="$GO_OUT_DIR" \
        --go_opt=paths=source_relative \
        --go-grpc_out="$GO_OUT_DIR" \
        --go-grpc_opt=paths=source_relative \
        "$proto_file"
    
    echo "  ✓ Generated Go code for $(basename "$proto_file")"
done

echo "✓ All Go protobuf files generated successfully!"
echo "Generated files in: $GO_OUT_DIR"
ls -la "$GO_OUT_DIR"