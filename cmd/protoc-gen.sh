#!/bin/bash

# Generate Go protobuf files
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    protos/stream_process.proto

# Generate gRPC-Gateway files
protoc --grpc-gateway_out=. --grpc-gateway_opt=paths=source_relative \
    --grpc-gateway_opt=generate_unbound_methods=true \
    protos/stream_process.proto

echo "Go protobuf files generated successfully"