"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""

    # 2023-10-30
    LLVM_COMMIT = "896749aa0d420ae573255a64a349bc2a76cfed37"

    new_git_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )
