get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    AffineFullUnroll
    MLIRNoisy
    MLIROptLib
    MLIRPass
    MLIRPoly
    MulToAdd
    NoisyPasses
    PolyToStandard
    ortools::ortools
)

add_llvm_executable(tutorial-opt tutorial-opt.cpp)

llvm_update_compile_flags(tutorial-opt)
target_link_libraries(tutorial-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tutorial-opt)
