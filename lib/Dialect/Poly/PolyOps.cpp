#include "lib/Dialect/Poly/PolyOps.h"

namespace mlir {
namespace tutorial {
namespace poly {

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir
