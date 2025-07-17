"""Configure LLVM Bazel overlays from a 'raw' imported llvm repository"""	

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")	

def setup_llvm(name):	
    """Build @llvm-project from @llvm-raw using the upstream bazel overlays."""	
    llvm_configure(name = name)
