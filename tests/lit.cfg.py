import os
from pathlib import Path

from lit.formats import ShTest

# oddly, the `config` variable is defined in the context of the lit runner that
# runs this module.

config.name = "mlir_tutorial"
config.test_format = ShTest()
config.suffixes = [".mlir"]

# lit executes relative to the directory
#
#   bazel-bin/tests/<test_target_name>.runfiles/mlir_tutorial/
#
# which contains all the binary targets included in via the `data` attribute in
# the lit.bzl macro, which in turn gets them from the filegroup //tests:test_utilities.
# To manually inspect what is included in the filesystem in situ, add the
# following to this script and run `bazel test //tests:<target>`
#
#   import subprocess
#
#   print(subprocess.run(["pwd",]).stdout)
#   print(subprocess.run(["ls", "-l", os.environ["RUNFILES_DIR"]]).stdout)
#   print(subprocess.run([ "env", ]).stdout)
#
# Bazel defines RUNFILES_DIR which includes mlir_tutorial/ and third party
# dependencies as their own directory. Generally, it seems that $PWD ==
# $RUNFILES_DIR/mlir_tutorial/
runfiles_dir = Path(os.environ["RUNFILES_DIR"])
tool_relpaths = [
    "llvm-project/mlir",
    "llvm-project/llvm",
    "mlir_tutorial/tools",
]

config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + ":"
    + os.environ["PATH"]
)

substitutions = {
    "%project_source_dir": str(runfiles_dir.joinpath(Path('mlir_tutorial'))),
}
config.substitutions.extend(substitutions.items())
