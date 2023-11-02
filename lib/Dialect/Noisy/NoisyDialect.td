#ifndef LIB_DIALECT_NOISY_NOISYDIALECT_TD_
#define LIB_DIALECT_NOISY_NOISYDIALECT_TD_

include "mlir/IR/OpBase.td"

def Noisy_Dialect : Dialect {
  let name = "noisy";
  let summary = "A dialect for arithmetic on noisy i32s";

  let cppNamespace = "::mlir::tutorial::noisy";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_NOISY_NOISYDIALECT_TD_
