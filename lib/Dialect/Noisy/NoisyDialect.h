#ifndef LIB_DIALECT_NOISY_NOISYDIALECT_H_
#define LIB_DIALECT_NOISY_NOISYDIALECT_H_

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include "mlir/include/mlir/IR/DialectImplementation.h"

#include "lib/Dialect/Noisy/NoisyDialect.h.inc"


constexpr int INITIAL_NOISE = 12;
constexpr int MAX_NOISE = 26;

#endif  // LIB_DIALECT_NOISY_NOISYDIALECT_H_
