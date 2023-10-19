#ifndef LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
#define LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def PolyToStandard : Pass<"poly-to-standard"> {
  let summary = "Lower `poly` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `poly` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tutorial::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif  // LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
