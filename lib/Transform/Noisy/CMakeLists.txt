add_mlir_library(NoisyPasses
    ReduceNoiseOptimizer.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Noisy/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    MLIRNoisy
    MLIRNoisyPasses

    LINK_LIBS PUBLIC
    ReduceNoiseAnalysis
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Noisy)
add_public_tablegen_target(MLIRNoisyPasses)
add_mlir_doc(Passes NoisyPasses ./ -gen-pass-doc)
