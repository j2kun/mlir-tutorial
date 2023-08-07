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
        "//lib/Transform/Affine:Passes",
        "//lib/Transform/Arith:Passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
    ],
)
