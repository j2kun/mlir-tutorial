#ifndef LIB_DIALECT_NOISY_NOISYOPS_H_
#define LIB_DIALECT_NOISY_NOISYOPS_H_

#include "lib/Dialect/Noisy/NoisyDialect.h"
#include "lib/Dialect/Noisy/NoisyTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"         // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Noisy/NoisyOps.h.inc"

#endif // LIB_DIALECT_NOISY_NOISYOPS_H_
