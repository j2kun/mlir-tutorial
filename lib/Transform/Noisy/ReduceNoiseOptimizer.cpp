#include "lib/Transform/Noisy/ReduceNoiseOptimizer.h"

#include "lib/Analysis/ReduceNoiseAnalysis/ReduceNoiseAnalysis.h"
#include "lib/Dialect/Noisy/NoisyOps.h"
#include "lib/Dialect/Noisy/NoisyTypes.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"
#include "mlir/include/mlir/IR/Visitors.h"
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

  void runOnOperation() {
    Operation *module = getOperation();

    ReduceNoiseAnalysis analysis(module);

    // Use the int range analysis to confirm the noise is always below the
    // maximum.
    DataFlowSolver solver;
    // The IntegerRangeAnalysis depends on DeadCodeAnalysis, but this
    // dependence is not automatic and fails silently.
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      signalPassFailure();

    auto result = module->walk([&](Operation *op) {
      if (!llvm::isa<noisy::AddOp, noisy::SubOp, noisy::MulOp,
                     noisy::ReduceNoiseOp>(*op)) {
        return WalkResult::advance();
      }
      const dataflow::IntegerValueRangeLattice *opRange =
          solver.lookupState<dataflow::IntegerValueRangeLattice>(
              op->getResult(0));
      if (!opRange || opRange->getValue().isUninitialized()) {
        op->emitOpError()
            << "Found op without a set integer range; did the analysis fail?";
        return WalkResult::interrupt();
      }

      ConstantIntRanges range = opRange->getValue().getValue();
      if (range.umax().getZExtValue() > MAX_NOISE) {
        op->emitOpError() << "Found op after which the noise exceeds the "
                             "allowable maximum of "
                          << MAX_NOISE
                          << "; it was: " << range.umax().getZExtValue()
                          << "\n";
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace noisy
} // namespace tutorial
} // namespace mlir
