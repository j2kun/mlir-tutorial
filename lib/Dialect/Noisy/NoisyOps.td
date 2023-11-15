#ifndef LIB_DIALECT_NOISY_NOISYOPS_TD_
#define LIB_DIALECT_NOISY_NOISYOPS_TD_

include "NoisyDialect.td"
include "NoisyTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Noisy_BinOp<string mnemonic> : Op<Noisy_Dialect, mnemonic, [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferIntRangeInterface>
]> {
  let arguments = (ins Noisy_I32:$lhs, Noisy_I32:$rhs);
  let results = (outs Noisy_I32:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
}

def Noisy_AddOp : Noisy_BinOp<"add"> {
  let summary = "Addition operation between noisy ints. Adds noise.";
}

def Noisy_SubOp : Noisy_BinOp<"sub"> {
  let summary = "Subtraction operation between noisy ints. Adds noise.";
}

def Noisy_MulOp : Noisy_BinOp<"mul"> {
  let summary = "Multiplication operation between noisy ints. Multiplies noise.";
}

def Noisy_EncodeOp : Op<Noisy_Dialect, "encode", [
    Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>]> {
  let summary = "Encodes a noisy i32 from a small-width integer, injecting 12 bits of noise.";
  let arguments = (ins AnyIntOfWidths<[1, 2, 3, 4, 5]>:$input);
  let results = (outs Noisy_I32:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";
}

def Noisy_DecodeOp : Op<Noisy_Dialect, "decode", [Pure]> {
  let summary = "Decodes a noisy integer to a regular integer, failing if the noise is too high.";
  let arguments = (ins Noisy_I32:$input);
  let results = (outs AnyIntOfWidths<[1, 2, 3, 4, 5]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
}

def Noisy_ReduceNoiseOp : Op<Noisy_Dialect, "reduce_noise", [
    Pure, SameOperandsAndResultType,  DeclareOpInterfaceMethods<InferIntRangeInterface>]> {
  let summary = "Reduces the noise in a noisy integer to a fixed noise level. Expensive!";
  let arguments = (ins Noisy_I32:$input);
  let results = (outs Noisy_I32:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($output))";
}

#endif  // LIB_DIALECT_NOISY_NOISYOPS_TD_
