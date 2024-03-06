#include "lib/Transform/Affine/AffineAnalysisDebug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_AFFINEANALYSISDEBUG
#include "lib/Transform/Affine/Passes.h.inc"

// A pass that manually walks the IR
struct AffineAnalysisDebug : impl::AffineAnalysisDebugBase<AffineAnalysisDebug> {
  using AffineAnalysisDebugBase::AffineAnalysisDebugBase;

  void runOnOperation() {
    getOperation()->emitError("shell");
  }
};



} // namespace tutorial
} // namespace mlir
