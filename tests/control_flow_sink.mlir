// RUN: tutorial-opt -control-flow-sink %s | FileCheck %s

// Test that operations can be sunk.

// CHECK-LABEL: @test_simple_sink
func.func @test_simple_sink(%arg0: i1) -> !poly.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  %1 = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
  %p1 = poly.from_tensor %1 : tensor<3xi32> -> !poly.poly<10>
  // CHECK-NOT: poly.from_tensor
  // CHECK: scf.if
  %4 = scf.if %arg0 -> (!poly.poly<10>) {
    // CHECK: poly.from_tensor
    %2 = poly.mul %p0, %p0 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  // CHECK: else
  } else {
    // CHECK: poly.from_tensor
    %3 = poly.mul %p1, %p1 : !poly.poly<10>
    scf.yield %3 : !poly.poly<10>
  }
  return %4 : !poly.poly<10>
}
