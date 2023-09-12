// RUN: tutorial-opt -cse %s | FileCheck %s

// CHECK-LABEL: @test_simple_cse
func.func @test_simple_cse() -> !poly.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK: poly.from_tensor
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  // exactly one mul op
  // CHECK-NEXT: poly.mul
  // CHECK-NEXT: poly.add
  %2 = poly.mul %p0, %p0 : !poly.poly<10>
  %3 = poly.mul %p0, %p0 : !poly.poly<10>
  %4 = poly.add %2, %3 : !poly.poly<10>
  return %4 : !poly.poly<10>
}
