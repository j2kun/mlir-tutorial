package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ReduceNoiseAnalysis",
    srcs = ["ReduceNoiseAnalysis.cpp"],
    hdrs = ["ReduceNoiseAnalysis.h"],
    deps = [
        "//lib/Dialect/Noisy",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/linear_solver",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
