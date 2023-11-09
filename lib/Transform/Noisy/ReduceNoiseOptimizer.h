#ifndef LIB_TRANSFORM_NOISY_REDUCENOISEOPTIMIZER_H_
#define LIB_TRANSFORM_NOISY_REDUCENOISEOPTIMIZER_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {
namespace noisy {

#define GEN_PASS_DECL_REDUCENOISEOPTIMIZER
#include "lib/Transform/Noisy/Passes.h.inc"

}  // namespace noisy
}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_NOISY_REDUCENOISEOPTIMIZER_H_
