add_mlir_dialect(Poly poly)
add_mlir_doc(PolyDialect PolyDialect Poly/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS PolyPatterns.td)
mlir_tablegen(PolyCanonicalize.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRPolyCanonicalizationIncGen)

add_mlir_dialect_library(MLIRPoly
    PolyDialect.cpp
    PolyOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly

    LINK_LIBS PUBLIC
    )