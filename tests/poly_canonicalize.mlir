// RUN: tutorial-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @test_simple
func.func @test_simple() -> !poly.poly<10> {
  // CHECK: poly.constant dense<[2, 4, 6]>
  // CHECK-NEXT: return
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  %2 = poly.add %p0, %p0 : !poly.poly<10>
  %3 = poly.mul %p0, %p0 : !poly.poly<10>
  %4 = poly.add %2, %3 : !poly.poly<10>
  return %2 : !poly.poly<10>
}

// CHECK-LABEL: func.func @test_difference_of_squares
// CHECK-SAME: %[[x:.+]]: !poly.poly<3>,
// CHECK-SAME: %[[y:.+]]: !poly.poly<3>
func.func @test_difference_of_squares(
    %0: !poly.poly<3>, %1: !poly.poly<3>) -> !poly.poly<3> {
  // CHECK: %[[sum:.+]] = poly.add %[[x]], %[[y]]
  // CHECK: %[[diff:.+]] = poly.sub %[[x]], %[[y]]
  // CHECK: %[[mul:.+]] = poly.mul %[[sum]], %[[diff]]
  %2 = poly.mul %0, %0 : !poly.poly<3>
  %3 = poly.mul %1, %1 : !poly.poly<3>
  %4 = poly.sub %2, %3 : !poly.poly<3>
  %5 = poly.add %4, %4 : !poly.poly<3>
  return %5 : !poly.poly<3>
}

// CHECK-LABEL: func.func @test_difference_of_squares_other_uses
// CHECK-SAME: %[[x:.+]]: !poly.poly<3>,
// CHECK-SAME: %[[y:.+]]: !poly.poly<3>
func.func @test_difference_of_squares_other_uses(
    %0: !poly.poly<3>, %1: !poly.poly<3>) -> !poly.poly<3> {
  // The canonicalization does not occur because x_squared has a second use.
  // CHECK: %[[x_squared:.+]] = poly.mul %[[x]], %[[x]]
  // CHECK: %[[y_squared:.+]] = poly.mul %[[y]], %[[y]]
  // CHECK: %[[diff:.+]] = poly.sub %[[x_squared]], %[[y_squared]]
  // CHECK: %[[sum:.+]] = poly.add %[[diff]], %[[x_squared]]
  %2 = poly.mul %0, %0 : !poly.poly<3>
  %3 = poly.mul %1, %1 : !poly.poly<3>
  %4 = poly.sub %2, %3 : !poly.poly<3>
  %5 = poly.add %4, %2 : !poly.poly<3>
  return %5 : !poly.poly<3>
}
