#ifndef LIB_TRANSFORM_ARITH_MULTOADDPDLL_H_
#define LIB_TRANSFORM_ARITH_MULTOADDPDLL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL_MULTOADDPDLL
#include "lib/Transform/Arith/Passes.h.inc"

#include "lib/Transform/Arith/MulToAddPdll.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_ARITH_MULTOADDPDLL_H_
