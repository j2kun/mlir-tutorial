#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

void AffineFullUnrollPass::runOnOperation() {
  AffineForOp op = getOperation();
  op.emitError() << "converted op successfully";
  auto result = loopUnrollFull(op);
  if (failed(result)) {
    op.emitError() << "failed";
    signalPassFailure();
  }
}

} // namespace tutorial
} // namespace mlir
