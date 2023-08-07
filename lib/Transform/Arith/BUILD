# Passes that work with the Arith dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Arith",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ArithPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "MulToAdd",
    srcs = ["MulToAdd.cpp"],
    hdrs = ["MulToAdd.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Passes",
    hdrs = ["Passes.h"],
    deps = [
        ":MulToAdd",
        ":pass_inc_gen",
    ],
)
