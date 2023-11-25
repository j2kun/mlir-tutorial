# MLIR For Beginners

This is the code repository for a series of articles on the
[MLIR framework](https://mlir.llvm.org/) for building compilers.

## Articles

1.  [Build System (Getting Started)](https://jeremykun.com/2023/08/10/mlir-getting-started/)
2.  [Running and Testing a Lowering](https://jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/)
3.  [Writing Our First Pass](https://jeremykun.com/2023/08/10/mlir-writing-our-first-pass/)
4.  [Using Tablegen for Passes](https://jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)
5.  [Defining a New Dialect](https://jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/)
6.  [Using Traits](https://jeremykun.com/2023/09/07/mlir-using-traits/)
7.  [Folders and Constant Propagation](https://jeremykun.com/2023/09/11/mlir-folders/)
8.  [Verifiers](https://jeremykun.com/2023/09/13/mlir-verifiers/)
9.  [Canonicalizers and Declarative Rewrite Patterns](https://jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/)
10. [Dialect Conversion](https://jeremykun.com/2023/10/23/mlir-dialect-conversion/)
11. [Lowering through LLVM](https://jeremykun.com/2023/11/01/mlir-lowering-through-llvm/)
12. [A Global Optimization and Dataflow Analysis](https://jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis/)

## Bazel build

Bazel is one of two supported build systems for this tutorial. The other is
CMake. If you're unfamiliar with Bazel, you can read the tutorials at
[https://bazel.build/start](https://bazel.build/start). Familiarity with Bazel
is not required to build or test, but it is required to follow the articles in
the tutorial series and explained in the first article,
[Build System (Getting Started)](https://jeremykun.com/2023/08/10/mlir-getting-started/).
The CMake build is maintained, but was added at article 10 (Dialect Conversion)
and will not be explained in the articles.

### Prerequisites

Install Bazelisk via instructions at
[https://github.com/bazelbuild/bazelisk#installation](https://github.com/bazelbuild/bazelisk#installation).
This should create the `bazel` command on your system.

You should also have a modern C++ compiler on your system, either `gcc` or
`clang`, which Bazel will detect.

### Build and test

Run

```bash
bazel build ...:all
bazel test ...:all
```

## CMake build

CMake is one of two supported build systems for this tutorial. The other is
Bazel. If you're unfamiliar with CMake, you can read the tutorials at
[https://cmake.org/getting-started/](https://cmake.org/getting-started/). The
CMake build is maintained, but was added at article 10 (Dialect Conversion) and
will not be explained in the articles.

### Prerequisites

*   Make sure you have installed everything needed to build LLVM
    https://llvm.org/docs/GettingStarted.html#software
*   For this recipe Ninja is used so be sure to have it as well installed
    https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages

### Checking out the code

Checkout the tutorial including the LLVM dependency (submodules):

```bash
git clone --recurse-submodules https://github.com/j2kun/mlir-tutorial.git
cd mlir-tutorial
```

### Building dependencies

Note: The following steps are suitable for macOs and use ninja as building
system, they should not be hard to adapt for your environment.

*Build LLVM/MLIR*

```bash
#!/bin/sh

BUILD_SYSTEM=Ninja
BUILD_TAG=ninja
THIRDPARTY_LLVM_DIR=$PWD/externals/llvm-project
BUILD_DIR=$THIRDPARTY_LLVM_DIR/build
INSTALL_DIR=$THIRDPARTY_LLVM_DIR/install

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR

pushd $BUILD_DIR

cmake ../llvm -G $BUILD_SYSTEM \
      -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
      -DCMAKE_C_COMPILER="$(xcrun --find clang)" \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DLLVM_LOCAL_RPATH=$INSTALL_DIR/lib \
      -DLLVM_PARALLEL_COMPILE_JOBS=7 \
      -DLLVM_PARALLEL_LINK_JOBS=1 \
      -DLLVM_BUILD_EXAMPLES=OFF \
      -DLLVM_INSTALL_UTILS=ON \
      -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_CCACHE_BUILD=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DLLVM_ENABLE_PROJECTS='mlir' \
      -DDEFAULT_SYSROOT="$(xcrun --show-sdk-path)" \
      -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)"

cmake --build . --target check-mlir

popd
```

### Build and test

```bash
#!/bin/sh

BUILD_SYSTEM="Ninja"
BUILD_DIR=./build-`echo ${BUILD_SYSTEM}| tr '[:upper:]' '[:lower:]'`

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
pushd $BUILD_DIR

LLVM_BUILD_DIR=externals/llvm-project/build
cmake -G $BUILD_SYSTEM .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DBUILD_DEPS="ON" \
    -DBUILD_SHARED_LIBS="OFF" \
    -DCMAKE_BUILD_TYPE=Debug

popd

cmake --build $BUILD_DIR --target MLIRAffineFullUnrollPasses
cmake --build $BUILD_DIR --target MLIRMulToAddPasses
cmake --build $BUILD_DIR --target MLIRNoisyPasses
cmake --build $BUILD_DIR --target mlir-headers
cmake --build $BUILD_DIR --target mlir-doc
cmake --build $BUILD_DIR --target tutorial-opt
cmake --build $BUILD_DIR --target check-mlir-tutorial
```
