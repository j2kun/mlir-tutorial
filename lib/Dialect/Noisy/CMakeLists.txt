# Inlining `add_mlir_dialect(Noisy noisy)` commands so that
# we can custom name `*.inc` generated files.
set(LLVM_TARGET_DEFINITIONS NoisyOps.td)
mlir_tablegen(NoisyOps.h.inc -gen-op-decls)
mlir_tablegen(NoisyOps.cpp.inc -gen-op-defs)
mlir_tablegen(NoisyTypes.h.inc -gen-typedef-decls -typedefs-dialect=noisy)
mlir_tablegen(NoisyTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=noisy)
mlir_tablegen(NoisyDialect.h.inc -gen-dialect-decls -dialect=noisy)
mlir_tablegen(NoisyDialect.cpp.inc -gen-dialect-defs -dialect=noisy)
add_public_tablegen_target(MLIRNoisyOpsIncGen)
add_dependencies(mlir-headers MLIRNoisyOpsIncGen)

add_mlir_doc(NoisyDialect NoisyDialect Noisy/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRNoisy
    NoisyDialect.cpp
    NoisyOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Noisy
    )
