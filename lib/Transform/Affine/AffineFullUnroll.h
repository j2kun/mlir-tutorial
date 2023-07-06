#ifndef LIB_CONVERSION_AFFINE_AFFINEFULLUNROLL_H_
#define LIB_CONVERSION_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::affine::AffineForOp>> {
private:
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  StringRef getArgument() const final { return "affine-full-unroll"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // LIB_CONVERSION_AFFINE_AFFINEFULLUNROLL_H_
