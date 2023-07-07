# Passes that work with the Arith dialect

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "MulToAdd",
    srcs = ["MulToAdd.cpp"],
    hdrs = ["MulToAdd.h"],
    deps = [
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
