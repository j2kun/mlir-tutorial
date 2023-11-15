# Passes that work with the Noisy dialect

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
                "-name=Noisy",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "NoisyPasses.md",
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
    name = "ReduceNoiseOptimizer",
    srcs = ["ReduceNoiseOptimizer.cpp"],
    hdrs = [
        "Passes.h",
        "ReduceNoiseOptimizer.h",
    ],
    deps = [
        ":pass_inc_gen",
        "//lib/Analysis/ReduceNoiseAnalysis",
        "//lib/Dialect/Noisy",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Passes",
    hdrs = ["Passes.h"],
    deps = [
        ":ReduceNoiseOptimizer",
        ":pass_inc_gen",
    ],
)
