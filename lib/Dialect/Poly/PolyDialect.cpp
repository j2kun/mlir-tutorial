#include "lib/Dialect/Poly/PolyDialect.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "mlir/include/mlir/IR/Builders.h"

#include "lib/Dialect/Poly/PolyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Poly/PolyOps.cpp.inc"

namespace mlir {
namespace tutorial {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Poly/PolyOps.cpp.inc"
      >();
}

} // namespace poly
} // namespace tutorial
} // namespace mlir
