
Run the following to see the no-op change

```
bazel run tools:tutorial-opt -- --affine-full-unroll < $(pwd)/tests/affine_loop_unroll.mlir
```

Then the following to see it segfault

```
bazel run tools:tutorial-opt -- --pass-pipeline='any(affine-full-unroll)' < $(pwd)/tests/affine_loop_unroll.mlir
```

Error:

```
tutorial-opt: external/llvm-project/llvm/include/llvm/Support/Casting.h:578:
decltype(auto) llvm::cast(From*) [with To = mlir::affine::AffineForOp; From = mlir::Operation]:
Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```
