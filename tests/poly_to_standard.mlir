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

// CHECK-LABEL: test_lower_mul
// CHECK-SAME:  (%[[p0:.*]]: [[T:tensor<10xi32>]], %[[p1:.*]]: [[T]]) -> [[T]] {
// CHECK:    %[[cst:.*]] = arith.constant dense<0> : [[T]]
// CHECK:    %[[c0:.*]] = arith.constant 0 : index
// CHECK:    %[[c10:.*]] = arith.constant 10 : index
// CHECK:    %[[c1:.*]] = arith.constant 1 : index
// CHECK:    %[[outer:.*]] = scf.for %[[outer_iv:.*]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[outer_iter_arg:.*]] = %[[cst]]) -> ([[T]]) {
// CHECK:      %[[inner:.*]] = scf.for %[[inner_iv:.*]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[inner_iter_arg:.*]] = %[[outer_iter_arg]]) -> ([[T]]) {
// CHECK:        %[[index_sum:.*]] = arith.addi %arg2, %arg4
// CHECK:        %[[dest_index:.*]] = arith.remui %[[index_sum]], %[[c10]]
// CHECK-DAG:    %[[p0_extracted:.*]] = tensor.extract %[[p0]][%[[outer_iv]]]
// CHECK-DAG:    %[[p1_extracted:.*]] = tensor.extract %[[p1]][%[[inner_iv]]]
// CHECK:        %[[coeff_mul:.*]] = arith.muli %[[p0_extracted]], %[[p1_extracted]]
// CHECK:        %[[accum:.*]] = tensor.extract %[[inner_iter_arg]][%[[dest_index]]]
// CHECK:        %[[to_insert:.*]] = arith.addi %[[coeff_mul]], %[[accum]]
// CHECK:        %[[inserted:.*]] = tensor.insert %[[to_insert]] into %[[inner_iter_arg]][%[[dest_index]]]
// CHECK:        scf.yield %[[inserted]]
// CHECK:      }
// CHECK:      scf.yield %[[inner]]
// CHECK:    }
// CHECK:    return %[[outer]]
// CHECK:  }
func.func @test_lower_mul(%0 : !poly.poly<10>, %1 : !poly.poly<10>) -> !poly.poly<10> {
  %2 = poly.mul %0, %1: !poly.poly<10>
  return %2 : !poly.poly<10>
}


// CHECK-LABEL: test_lower_eval
// CHECK-SAME:  (%[[poly:.*]]: [[T:tensor<10xi32>]], %[[point:.*]]: i32) -> i32 {
// CHECK:    %[[c1:.*]] = arith.constant 1 : index
// CHECK:    %[[c10:.*]] = arith.constant 10 : index
// CHECK:    %[[c11:.*]] = arith.constant 11 : index
// CHECK:    %[[accum:.*]] = arith.constant 0 : i32
// CHECK:    %[[loop:.*]] = scf.for %[[iv:.*]] = %[[c1]] to %[[c11]] step %[[c1]] iter_args(%[[iter_arg:.*]] = %[[accum]]) -> (i32) {
// CHECK:        %[[coeffIndex:.*]] = arith.subi %[[c10]], %[[iv]]
// CHECK:        %[[mulOp:.*]] = arith.muli %[[point]], %[[iter_arg]]
// CHECK:        %[[nextCoeff:.*]] = tensor.extract %[[poly]][%[[coeffIndex]]]
// CHECK:        %[[next:.*]] = arith.addi %[[mulOp]], %[[nextCoeff]]
// CHECK:        scf.yield %[[next]]
// CHECK:    }
// CHECK:    return %[[loop]]
// CHECK:  }
func.func @test_lower_eval(%0 : !poly.poly<10>, %1 : i32) -> i32 {
  %2 = poly.eval %0, %1: (!poly.poly<10>, i32) -> i32
  return %2 : i32
}


// CHECK-LABEL: test_lower_many
// CHECK-NOT: poly
func.func @test_lower_many(%arg : !poly.poly<10>, %point : i32) -> i32 {
  %0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  %1 = poly.add %0, %arg : !poly.poly<10>
  %2 = poly.mul %1, %1 : !poly.poly<10>
  %3 = poly.sub %2, %arg : !poly.poly<10>
  %4 = poly.eval %3, %point: (!poly.poly<10>, i32) -> i32
  return %4 : i32
}
