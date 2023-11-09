#include "lib/Analysis/ReduceNoiseAnalysis/ReduceNoiseAnalysis.h"

#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project

namespace mlir {
namespace tutorial {

ReduceNoiseAnalysis::ReduceNoiseAnalysis(Operation *op) {
  // Implement analysis here
}

}  // namespace tutorial
}  // namespace mlir
