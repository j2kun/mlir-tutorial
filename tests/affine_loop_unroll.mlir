// RUN: tutorial-opt %s --affine-full-unroll > %t
// RUN: FileCheck %s < %t

// RUN: tutorial-opt %s --affine-full-unroll-rewrite > %t
// RUN: FileCheck %s < %t

func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  // CHECK-LABEL: test_single_nested_loop
  // CHECK-NOT: affine.for
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}

func.func @test_doubly_nested_loop(%buffer: memref<4x3xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  // CHECK-LABEL: test_doubly_nested_loop
  // CHECK-NOT: affine.for
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %sum_nested_0 = arith.constant 0 : i32
    %sum_nested = affine.for %j = 0 to 3 iter_args(%sum_nested_iter = %sum_nested_0) -> i32 {
      %t = affine.load %buffer[%i, %j] : memref<4x3xi32>
      %sum_nested_next = arith.addi %sum_nested_iter, %t : i32
      affine.yield %sum_nested_next : i32
    }
    %sum_next = arith.addi %sum_iter, %sum_nested : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
