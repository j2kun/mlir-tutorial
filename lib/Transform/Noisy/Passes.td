#ifndef LIB_TRANSFORM_NOISY_PASSES_TD_
#define LIB_TRANSFORM_NOISY_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ReduceNoiseOptimizer : Pass<"noisy-reduce-noise-optimizer"> {
  let summary = "Insert reduce_noise ops optimally";
  let description = [{
    Solves an integer linear program to select the optimal locations in the IR
    to insert `reduce_noise` ops.
  }];
  let dependentDialects = ["mlir::tutorial::noisy::NoisyDialect"];
}

#endif  // LIB_TRANSFORM_NOISY_PASSES_TD_
