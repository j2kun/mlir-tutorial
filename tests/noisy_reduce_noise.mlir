// RUN: tutorial-opt %s | FileCheck %s
// Check for syntax

// CHECK-LABEL: test_op_syntax
func.func @test_op_syntax() -> i5 {
  %0 = arith.constant 3 : i5
  %1 = arith.constant 4 : i5
  %2 = noisy.encode %0 : i5 -> !noisy.i32
  %3 = noisy.encode %1 : i5 -> !noisy.i32
  %4 = noisy.mul %2, %3 : !noisy.i32
  %5 = noisy.mul %4, %4 : !noisy.i32
  %6 = noisy.mul %5, %5 : !noisy.i32
  %7 = noisy.mul %6, %6 : !noisy.i32
  %8 = noisy.decode %7 : !noisy.i32 -> i5
  return %8 : i5
}
