// RUN: tutorial-opt --poly-to-standard %s | FileCheck %s

// CHECK-LABEL: test_lower_add
func.func @test_lower_add(%0 : !poly.poly<10>, %1 : !poly.poly<10>) -> !poly.poly<10> {
  // CHECK: arith.addi
  %2 = poly.add %0, %1: !poly.poly<10>
  return %2 : !poly.poly<10>
}

// CHECK-LABEL: test_lower_sub
func.func @test_lower_sub(%0 : !poly.poly<10>, %1 : !poly.poly<10>) -> !poly.poly<10> {
  // CHECK: arith.subi
  %2 = poly.sub %0, %1: !poly.poly<10>
  return %2 : !poly.poly<10>
}

// CHECK-LABEL: test_lower_to_tensor(
// CHECK-SAME: %[[V0:.*]]: [[T:tensor<10xi32>]]) -> [[T]] {
// CHECK-NEXT:    return %[[V0]] : [[T]]
func.func @test_lower_to_tensor(%0: !poly.poly<10>) -> tensor<10xi32> {
  %2 = poly.to_tensor %0: !poly.poly<10> -> tensor<10xi32>
  return %2 : tensor<10xi32>
}

// CHECK-LABEL: test_lower_from_tensor(
// CHECK-SAME: %[[V0:.*]]: [[T:tensor<10xi32>]]) -> [[T]] {
// CHECK-NEXT:    return %[[V0]] : [[T]]
func.func @test_lower_from_tensor(%0 : tensor<10xi32>) -> !poly.poly<10> {
  %2 = poly.from_tensor %0: tensor<10xi32> -> !poly.poly<10>
  return %2 : !poly.poly<10>
}

// CHECK-LABEL: test_lower_from_tensor_extend(
// CHECK-SAME: %[[V0:.*]]: [[T:tensor<10xi32>]]) -> [[T2:tensor<20xi32>]] {
// CHECK:         %[[V1:.*]] = tensor.pad %[[V0]] low[0] high[10]
// CHECK:         return %[[V1]] : [[T2]]
func.func @test_lower_from_tensor_extend(%0 : tensor<10xi32>) -> !poly.poly<20> {
  %2 = poly.from_tensor %0: tensor<10xi32> -> !poly.poly<20>
  return %2 : !poly.poly<20>
}

// CHECK-LABEL: test_lower_add_and_fold
func.func @test_lower_add_and_fold() {
  // CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi32>
  %0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  // CHECK: arith.constant dense<[3, 4, 5]> : tensor<3xi32>
  %1 = poly.constant dense<[3, 4, 5]> : tensor<3xi32> : !poly.poly<10>
  // would be an addi, but it was folded
  // CHECK: arith.constant
  %2 = poly.add %0, %1: !poly.poly<10>
  return
}
