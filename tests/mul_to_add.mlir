// RUN: tutorial-opt %s --mul-to-add > %t
// RUN: FileCheck %s < %t

func.func @just_power_of_two(%arg: i32) -> i32 {
  %0 = arith.constant 8 : i32
  %1 = arith.muli %arg, %0 : i32
  func.return %1 : i32
}

// CHECK-LABEL: func.func @just_power_of_two(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:   %[[SUM_0:.*]] = arith.addi %[[ARG]], %[[ARG]]
// CHECK:   %[[SUM_1:.*]] = arith.addi %[[SUM_0]], %[[SUM_0]]
// CHECK:   %[[SUM_2:.*]] = arith.addi %[[SUM_1]], %[[SUM_1]]
// CHECK:   return %[[SUM_2]] : i32
// CHECK: }


func.func @power_of_two_plus_one(%arg: i32) -> i32 {
  %0 = arith.constant 9 : i32
  %1 = arith.muli %arg, %0 : i32
  func.return %1 : i32
}

// CHECK-LABEL: func.func @power_of_two_plus_one(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:   %[[SUM_0:.*]] = arith.addi %[[ARG]], %[[ARG]]
// CHECK:   %[[SUM_1:.*]] = arith.addi %[[SUM_0]], %[[SUM_0]]
// CHECK:   %[[SUM_2:.*]] = arith.addi %[[SUM_1]], %[[SUM_1]]
// CHECK:   %[[SUM_3:.*]] = arith.addi %[[SUM_2]], %[[ARG]]
// CHECK:   return %[[SUM_3]] : i32
// CHECK: }
