add_mlir_library(AffinePasses
    AffineAnalysisDebug.cpp
    AffineFullUnroll.cpp
    AffineFullUnrollPatternRewrite.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    MLIRAffinePasses

    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRAffinePasses)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
