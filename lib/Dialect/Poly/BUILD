load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "PolyDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "PolyDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyDialect.td",
    deps = [
        # the base mlir target for defining operations and dialects in tablegen
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "Poly",
    srcs = ["PolyDialect.cpp"],
    hdrs = ["PolyDialect.h"],
    deps = [
        ":dialect_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)
