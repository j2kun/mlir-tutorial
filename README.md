## Install LLVM and MLIR jointly and set up environment

From the MLIR [setup page](https://mlir.llvm.org/getting_started/), from any
base directory of your choice on your system (not within this project).

Note, requires `clang`, and I used Debian clang version 14.0.6, as well as
`lld`. Run `sudo apt install clang lld`

```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-mlir
```

Then, back in this project:

Copy `env.template` to `.env` with the path to the `build` directory created in
the last command as `LLVM_BUILD_DIR`. Then run `source .env`.

```
# .env
LLVM_BUILD_DIR=/home/jkun/llvm-project/build
```

And run

```bash
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-tutorial
```
