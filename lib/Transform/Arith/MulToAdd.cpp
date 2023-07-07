#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand :
  public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

// Replace y = 9*x with y = 8*x + x
struct PeelFromMul :
  public OpRewritePattern<MulIOp> {
  PeelFromMul(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

void MulToAddPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<PowerOfTwoExpand>(&getContext());
  patterns.add<PeelFromMul>(&getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace tutorial
} // namespace mlir
