load("//bazel:lit.bzl", "glob_lit_tests")

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tests:lit.cfg.py",
        "//tools:tutorial-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:mlir-cpu-runner",
        "@llvm-project//mlir:mlir-opt",
        "@mlir_tutorial_pip_deps_lit//:pkg",
    ],
)

glob_lit_tests()
