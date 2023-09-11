#ifndef LIB_DIALECT_POLY_POLYOPS_H_
#define LIB_DIALECT_POLY_POLYOPS_H_

#include "lib/Dialect/Poly/PolyDialect.h"
#include "lib/Dialect/Poly/PolyTraits.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"         // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Poly/PolyOps.h.inc"

#endif // LIB_DIALECT_POLY_POLYOPS_H_
