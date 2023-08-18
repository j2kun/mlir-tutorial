"""Configure LLVM Bazel overlays from a 'raw' imported llvm repository"""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

# The subset of LLVM backend targets that should be compiled
_LLVM_TARGETS = [
    "X86",
    # The bazel dependency graph for mlir-opt fails to load (at the analysis
    # step) without the NVPTX target in this list, because mlir/test:TestGPU
    # depends on the //llvm:NVPTXCodeGen target, which is not defined unless this
    # is included. @j2kun asked the LLVM maintiners for tips on how to fix this,
    # see https://github.com/llvm/llvm-project/issues/63135
    "NVPTX",
    # Needed for Apple M1 targets, see
    # https://github.com/j2kun/mlir-tutorial/issues/11
    "AArch64",
]

def setup_llvm(name):
    """Build @llvm-project from @llvm-raw using the upstream bazel overlays."""
    llvm_configure(
        name = name,
        targets = _LLVM_TARGETS,
    )
