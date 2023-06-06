workspace(name = "mlir_tutorial")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# Bazel skylib is a collection of bazel tools that are a required dependency of
# the LLVM/MLIR bazel build overlay.
http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# A two-step process for buliding LLVM/MLIR with bazel. First the raw source
# code is downloaded and imported into this workspace as a git repository,
# called `llvm-raw`. Then the build files defined in the LLVM monorepo are
# overlaid using llvm_configure in the setup script below. This defines the
# @llvm-project bazel project which is can be built and depended on.
load("//bazel:import_llvm.bzl", "import_llvm")

import_llvm("llvm-raw")

load("//bazel:setup_llvm.bzl", "setup_llvm")

setup_llvm("llvm-project")
