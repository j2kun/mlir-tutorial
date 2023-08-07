#ifndef LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLLPATTERNREWRITE_H_
#define LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLLPATTERNREWRITE_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL_AFFINEFULLUNROLLPATTERNREWRITE
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLLPATTERNREWRITE_H_
