#!/bin/bash
# Fix Go import paths to use local module path

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Fixing Go import paths..."
echo "Project root: $PROJECT_ROOT"

# Find all Go files
GO_FILES=($(find "$PROJECT_ROOT" -name "*.go" -not -path "*/vendor/*"))

echo "Found ${#GO_FILES[@]} Go files to update"

# Replace GitHub import paths with local module paths
for go_file in "${GO_FILES[@]}"; do
    echo "Updating imports in $(basename "$go_file")..."
    
    # Replace the old module import paths
    sed -i '' 's|github\.com/streamprocess/streamprocess/protos|streamprocess/pkg/protos|g' "$go_file"
    sed -i '' 's|github\.com/streamprocess/streamprocess/pkg/queue|streamprocess/pkg/queue|g' "$go_file"
    sed -i '' 's|github\.com/streamprocess/streamprocess/pkg/|streamprocess/pkg/|g' "$go_file"
    
    echo "  ✓ Updated $(basename "$go_file")"
done

echo "✓ All Go import paths updated successfully!"