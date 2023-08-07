# Passes that work with the Affine dialect

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
                "-name=Affine",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "AffinePasses.md",
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
    name = "AffineFullUnroll",
    srcs = ["AffineFullUnroll.cpp"],
    hdrs = [
        "AffineFullUnroll.h",
        "Passes.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "AffineFullUnrollPatternRewrite",
    srcs = ["AffineFullUnrollPatternRewrite.cpp"],
    hdrs = [
        "AffineFullUnrollPatternRewrite.h",
        "Passes.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Passes",
    hdrs = ["Passes.h"],
    deps = [
        ":AffineFullUnroll",
        ":AffineFullUnrollPatternRewrite",
        ":pass_inc_gen",
    ],
)
