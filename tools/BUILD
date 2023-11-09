# The "tools" directory contains binary targets that expose the public API of
# the passes in the project.

package(
    default_visibility = ["//visibility:public"],
)

# We name the tool `tutorial-opt` following the pattern of `mlir-opt`.
cc_binary(
    name = "tutorial-opt",
    srcs = ["tutorial-opt.cpp"],
    includes = ["include"],
    deps = [
        "//lib/Conversion/PolyToStandard",
        "//lib/Dialect/Noisy",
        "//lib/Dialect/Poly",
        "//lib/Transform/Affine:Passes",
        "//lib/Transform/Arith:Passes",
        "//lib/Transform/Noisy:Passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BufferizationPipelines",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TensorToLinalg",
    ],
)
