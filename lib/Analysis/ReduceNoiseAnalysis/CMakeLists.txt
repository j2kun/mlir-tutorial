add_mlir_library(ReduceNoiseAnalysis
    ReduceNoiseAnalysis.cpp

    ${PROJECT_SOURCE_DIR}/lib/Analysis/ReduceNoiseAnalysis/
    ADDITIONAL_HEADER_DIRS
    LINK_LIBS PUBLIC
    ortools::ortools
)
