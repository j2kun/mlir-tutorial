#ifndef LIB_DIALECT_POLY_POLYOPS_TD_
#define LIB_DIALECT_POLY_POLYOPS_TD_

include "PolyDialect.td"
include "PolyTypes.td"

def Poly_AddOp : Op<Poly_Dialect, "add"> {
  let summary = "Addition operation between polynomials.";
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

#endif  // LIB_DIALECT_POLY_POLYOPS_TD_
