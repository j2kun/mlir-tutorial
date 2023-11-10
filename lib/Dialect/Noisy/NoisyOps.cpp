#include "lib/Dialect/Noisy/NoisyOps.h"

namespace mlir {
namespace tutorial {
namespace noisy {

ConstantIntRanges initialNoiseRange() {
  return ConstantIntRanges::fromUnsigned(APInt(32, 0),
                                         APInt(32, INITIAL_NOISE));
}

ConstantIntRanges unionPlusOne(ArrayRef<ConstantIntRanges> inputRanges) {
  auto lhsRange = inputRanges[0];
  auto rhsRange = inputRanges[1];
  auto joined = lhsRange.rangeUnion(rhsRange);
  return ConstantIntRanges::fromUnsigned(joined.umin(), joined.umax() + 1);
}

void EncodeOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                 SetIntRangeFn setResultRange) {
  setResultRange(getResult(), initialNoiseRange());
}

void AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), unionPlusOne(inputRanges));
}

void SubOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), unionPlusOne(inputRanges));
}

void MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                              SetIntRangeFn setResultRange) {
  auto lhsRange = inputRanges[0];
  auto rhsRange = inputRanges[1];
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(
                                  lhsRange.umin() + rhsRange.umin(),
                                  lhsRange.umax() + rhsRange.umax()));
}

void ReduceNoiseOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), initialNoiseRange());
}

} // namespace noisy
} // namespace tutorial
} // namespace mlir
