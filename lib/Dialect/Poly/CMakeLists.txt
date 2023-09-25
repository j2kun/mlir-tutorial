add_mlir_dialect(Poly poly)
add_mlir_doc(PolyDialect PolyDialect Poly/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRPoly
    PolyDialect.cpp
    PolyOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly

    LINK_LIBS PUBLIC
    )