# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLIR (Multi-Level Intermediate Representation) tutorial codebase demonstrating compiler construction concepts. The project implements custom MLIR dialects and passes, with comprehensive examples showing how to build compiler transformations.

## Build Systems

The project supports two build systems:

### Bazel (Primary)
- **Version**: Bazel 8.3.1 with Bzlmod enabled
- **Dependency Management**: Uses MODULE.bazel with Bazel Central Registry (BCR)
- **Build all**: `bazel build ...:all`
- **Test all**: `bazel test ...:all`
- **Build specific target**: `bazel build //tools:tutorial-opt`
- **Test specific target**: `bazel test //tests:poly_syntax`

#### Bzlmod Migration
The project has been migrated from WORKSPACE-based dependency management to Bzlmod:
- **MODULE.bazel**: Main module definition with dependencies from BCR
- **extensions.bzl**: Custom module extension for LLVM-related dependencies
- **WORKSPACE**: Simplified to workspace name only
- Most dependencies (rules_python, protobuf, or-tools, etc.) now use BCR
- LLVM dependencies still use git repositories for latest upstream integration

### CMake (Secondary)
First build LLVM/MLIR dependencies, then:
- **Configure**: `cmake -G Ninja -DLLVM_DIR="externals/llvm-project/build/lib/cmake/llvm" -DMLIR_DIR="externals/llvm-project/build/lib/cmake/mlir" -DBUILD_DEPS=ON -DCMAKE_BUILD_TYPE=Debug .`
- **Build main tool**: `cmake --build build-ninja --target tutorial-opt`
- **Run tests**: `cmake --build build-ninja --target check-mlir-tutorial`

## Key Architecture

### Core Components
- **tutorial-opt**: Main compiler tool in `tools/tutorial-opt.cpp` - processes MLIR files through various passes
- **Custom Dialects**: 
  - `Poly`: Polynomial arithmetic dialect (`lib/Dialect/Poly/`)
  - `Noisy`: Demonstration dialect with noise operations (`lib/Dialect/Noisy/`)
- **Transforms**: Located in `lib/Transform/` with subdirectories for different pass categories
- **Conversions**: Dialect lowering passes in `lib/Conversion/`
- **Analysis**: Data flow analysis passes in `lib/Analysis/`

### Test Infrastructure
- Uses LLVM's `lit` testing framework
- Test files in `tests/` directory with `.mlir` extension
- Run individual tests: `bazel test //tests:test_name`

### Pass Pipeline
The `poly-to-llvm` pipeline demonstrates full lowering:
1. Poly dialect → Standard dialect
2. Standard → Linalg
3. Bufferization
4. Linalg → Loops
5. Loops → Control Flow
6. Control Flow → LLVM IR

## Development Commands

### Testing Individual Components
- **Syntax test**: `tutorial-opt tests/poly_syntax.mlir`
- **Canonicalization**: `tutorial-opt --canonicalize tests/poly_canonicalize.mlir`
- **Full lowering**: `tutorial-opt --poly-to-llvm tests/poly_to_llvm.mlir`

### Adding New Passes
1. Define in appropriate `lib/Transform/*/Passes.td`
2. Implement in corresponding `.cpp` file
3. Register in `tools/tutorial-opt.cpp`
4. Add test in `tests/`

### Tablegen Files
- Dialect definitions: `*.td` files in `lib/Dialect/*/`
- Pass definitions: `Passes.td` files in `lib/Transform/*/`
- Pattern definitions: `*.td` files for rewrite patterns

## File Structure
- `lib/`: Core implementation (dialects, passes, conversions, analysis)
- `tools/`: Command-line tools (mainly tutorial-opt)
- `tests/`: Test files using lit framework
- `bazel/`: Bazel build configuration (legacy, LLVM import utilities)
- `externals/`: LLVM/MLIR submodule (for CMake builds)
- `MODULE.bazel`: Bzlmod module definition with BCR dependencies
- `extensions.bzl`: Custom module extension for LLVM dependencies
- `WORKSPACE`: Simplified workspace marker (legacy dependencies removed)
- `.bazelversion`: Pins Bazel version to 8.3.1
- `.bazelrc`: Build configuration with Bzlmod enabled