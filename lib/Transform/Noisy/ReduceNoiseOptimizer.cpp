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

struct ReduceNoiseOptimizer
    : impl::ReduceNoiseOptimizerBase<ReduceNoiseOptimizer> {
  using ReduceNoiseOptimizerBase::ReduceNoiseOptimizerBase;

  void runOnOperation() {
    Operation *module = getOperation();

    // FIXME: Should have some way to mark failure when solver is infeasible
    ReduceNoiseAnalysis analysis(module);
    OpBuilder b(&getContext());

    module->walk([&](Operation *op) {
      if (!analysis.shouldInsertReduceNoise(op))
        return;

      b.setInsertionPointAfter(op);
      auto reduceOp = b.create<ReduceNoiseOp>(op->getLoc(), op->getResult(0));
      op->getResult(0).replaceAllUsesExcept(reduceOp.getResult(), {reduceOp});
    });

    // Use the int range analysis to confirm the noise is always below the
    // maximum.
    DataFlowSolver solver;
    // The IntegerRangeAnalysis depends on DeadCodeAnalysis, but this
    // dependence is not automatic and fails silently.
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

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

    if (result.wasInterrupted()) {
      getOperation()->emitOpError()
          << "Detected error in the noise analysis.\n";
      signalPassFailure();
    }
  }
};

} // namespace noisy
} // namespace tutorial
} // namespace mlir
