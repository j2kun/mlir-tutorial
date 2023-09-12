// RUN: tutorial-opt %s 2>%t; FileCheck %s < %t

func.func @test_invalid_evalop(%arg0: !poly.poly<10>, %cst: i64) -> i64 {
  // This is a little brittle, since it matches both the error message
  // emitted by Has32BitArguments as well as that of EvalOp::verify.
  // I manually tested that they both fire when the input is as below.
  // CHECK: to be a 32-bit integer
  %0 = poly.eval %arg0, %cst : (!poly.poly<10>, i64) -> i64
  return %0 : i64
}
