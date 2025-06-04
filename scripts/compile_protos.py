#!/usr/bin/env python3
"""Compile Protocol Buffer definitions."""

import os
import sys
import subprocess
from pathlib import Path


def compile_protos():
    """Compile .proto files to Python modules."""
    # Get project root
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "protos"
    output_dir = project_root / "src" / "generated"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py in generated directory
    (output_dir / "__init__.py").touch()
    
    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print("No .proto files found in", proto_dir)
        return False
    
    print(f"Found {len(proto_files)} proto file(s)")
    
    for proto_file in proto_files:
        print(f"Compiling {proto_file.name}...")
        
        # Compile with protoc
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✓ Successfully compiled {proto_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to compile {proto_file.name}")
            print(f"    Error: {e.stderr}")
            return False
    
    # Fix imports in generated files (grpc_tools generates absolute imports)
    print("\nFixing imports in generated files...")
    for py_file in output_dir.glob("*_pb2*.py"):
        with open(py_file, "r") as f:
            content = f.read()
        
        # Replace absolute imports with relative imports
        if "_pb2_grpc.py" in str(py_file):
            content = content.replace(
                "import stream_process_pb2",
                "from . import stream_process_pb2"
            )
        
        with open(py_file, "w") as f:
            f.write(content)
    
    print("✓ All proto files compiled successfully!")
    return True


if __name__ == "__main__":
    success = compile_protos()
    sys.exit(0 if success else 1)