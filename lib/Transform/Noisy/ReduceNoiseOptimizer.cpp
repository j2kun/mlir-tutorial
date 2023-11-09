#include "lib/Transform/Noisy/ReduceNoiseOptimizer.h"

#include "lib/Dialect/Noisy/NoisyOps.h"
#include "lib/Dialect/Noisy/NoisyTypes.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {
namespace noisy {

#define GEN_PASS_DEF_REDUCENOISEOPTIMIZER
#include "lib/Transform/Noisy/Passes.h.inc"

// A pass that manually walks the IR
struct ReduceNoiseOptimizer
    : impl::ReduceNoiseOptimizerBase<ReduceNoiseOptimizer> {
  using ReduceNoiseOptimizerBase::ReduceNoiseOptimizerBase;

  void runOnOperation() { ; }
};

} // namespace noisy
} // namespace tutorial
} // namespace mlir
