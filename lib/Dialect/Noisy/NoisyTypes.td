#ifndef LIB_DIALECT_NOISY_NOISYTYPES_TD_
#define LIB_DIALECT_NOISY_NOISYTYPES_TD_

include "NoisyDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Noisy_Type<string name, string typeMnemonic> : TypeDef<Noisy_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Noisy_I32 : Noisy_Type<"NoisyI32", "i32"> {
  let summary = "A type for approximate 32-bit integers.";
}

#endif  // LIB_DIALECT_NOISY_NOISYTYPES_TD_
