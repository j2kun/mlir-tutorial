#include "lib/Transform/Arith/MulToAddPdll.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_MULTOADDPDLL
#include "lib/Transform/Arith/Passes.h.inc"


LogicalResult halveImpl(PatternRewriter &rewriter, PDLResultList &results,
                        ArrayRef<PDLValue> args) {
  Attribute attr = args[0].cast<Attribute>();
  IntegerAttr cAttr = cast<IntegerAttr>(attr);
  int64_t value = cAttr.getValue().getSExtValue();
  results.push_back(rewriter.getIntegerAttr(cAttr.getType(), value / 2));
  return success();
}

void registerNativeConstraints(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("Halve", halveImpl);
}

struct MulToAddPdll : impl::MulToAddPdllBase<MulToAddPdll> {
  using MulToAddPdllBase::MulToAddPdllBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    registerNativeConstraints(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace tutorial
} // namespace mlir
