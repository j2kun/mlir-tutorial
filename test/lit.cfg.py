# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'STANDALONE'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tutorial_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
#
# j2kun note: Sadly, this is not documented in the lit documentation, so it's
# not clear why individual files are included in this list of directories,
# nor why it matters if the suffixes list above is specified as `.mlir`
config.excludes = [
    'Inputs',
    'Examples',
    'CMakeLists.txt',
    'README.txt',
    'LICENSE.txt',
]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tutorial_obj_root, 'test')
config.tutorial_tools_dir = os.path.join(config.tutorial_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.tutorial_tools_dir, config.llvm_tools_dir]
tools = [
    # 'tutorial-opt',
    # 'tutorial-translate',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    'PYTHONPATH',
    [
        os.path.join(config.mlir_obj_dir, 'python_packages', 'tutorial'),
    ],
    append_path=True,
)
