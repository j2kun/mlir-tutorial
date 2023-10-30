#ifndef LIB_DIALECT_POLY_POLYPATTERNS_TD_
#define LIB_DIALECT_POLY_POLYPATTERNS_TD_

include "PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;

#endif  // LIB_DIALECT_POLY_POLYPATTERNS_TD_
