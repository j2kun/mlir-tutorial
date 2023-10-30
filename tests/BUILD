load("//bazel:lit.bzl", "glob_lit_tests")

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tests:lit.cfg.py",
        "//tests:poly_to_llvm_main.c",
        "//tools:tutorial-opt",
        "@llvm-project//clang:clang",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:llc",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:mlir-cpu-runner",
        "@llvm-project//mlir:mlir-opt",
        "@llvm-project//mlir:mlir-translate",
        "@mlir_tutorial_pip_deps_lit//:pkg",
    ],
)

glob_lit_tests()
