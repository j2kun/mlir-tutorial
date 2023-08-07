"""Macros for defining lit tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_python//python:py_test.bzl", "py_test")

_DEFAULT_FILE_EXTS = ["mlir"]

def lit_test(name = None, src = None, size = "small", tags = None):
    """Define a lit test.

    In its simplest form, a manually defined lit test would look like this:

      py_test(
          name = "ops.mlir.test",
          srcs = ["@llvm_project//llvm:lit"],
          args = ["-v", "tests/ops.mlir"],
          data = [":test_utilities", ":ops.mlir"],
          size = "small",
          main = "lit.py",
      )

    Where the `ops.mlir` file contains the test cases in standard RUN + CHECK
    format.

    The adjacent :test_utilities target contains all the tools (like mlir-opt)
    and files (like lit.cfg.py) that are needed to run a lit test. lit.cfg.py
    further specifies the lit configuration, including augmenting $PATH to
    include any mlir-opt-like tools.

    This macro simplifies the above definition by filling in the boilerplate.

    Args:
      name: the name of the test.
      src: the source file for the test.
      size: the size of the test.
      tags: tags to pass to the target.
    """
    if not src:
        fail("src must be specified")
    name = name or src + ".test"

    filegroup_name = name + ".filegroup"
    native.filegroup(
        name = filegroup_name,
        srcs = [src],
    )

    py_test(
        name = name,
        size = size,
        # -v ensures lit outputs useful info during test failures
        args = ["-v", paths.join(native.package_name(), src)],
        data = ["@mlir_tutorial//tests:test_utilities", filegroup_name],
        srcs = ["@llvm-project//llvm:lit"],
        main = "lit.py",
        tags = tags,
    )

def glob_lit_tests():
    """Searches the caller's directory for files to run as lit tests."""
    tests = native.glob(["*.mlir"])
    for curr_test in tests:
        lit_test(src = curr_test, size = "small")
