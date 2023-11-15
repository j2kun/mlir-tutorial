// RUN: tutorial-opt %s --noisy-reduce-noise-optimizer | FileCheck %s
// Check for syntax

// CHECK-LABEL: test_insert_noise_reduction_ops_mul
// CHECK:      [[V0:%.*]] = arith.constant 3
// CHECK-NEXT: [[V1:%.*]] = arith.constant 4
// CHECK-NEXT: [[V2:%.*]] = noisy.encode [[V0]]
// CHECK-NEXT: [[V3:%.*]] = noisy.encode [[V1]]
// CHECK-NEXT: [[V4:%.*]] = noisy.mul [[V2]], [[V3]]
// CHECK-NEXT: [[V4_R:%.*]] = noisy.reduce_noise [[V4]]
// CHECK-NEXT: [[V5:%.*]] = noisy.mul [[V4_R]], [[V4_R]]
// CHECK-NEXT: [[V5_R:%.*]] = noisy.reduce_noise [[V5]]
// CHECK-NEXT: [[V6:%.*]] = noisy.mul [[V5_R]], [[V5_R]]
// CHECK-NEXT: [[V6_R:%.*]] = noisy.reduce_noise [[V6]]
// This last mul does not need to be reduced
// CHECK-NEXT: [[V7:%.*]] = noisy.mul [[V6_R]], [[V6_R]]
// CHECK-NEXT: [[V8:%.*]] = noisy.decode [[V7]]
// CHECK-NEXT: return
func.func @test_insert_noise_reduction_ops_mul() -> i5 {
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

// CHECK-LABEL: test_insert_noise_reduction_ops_add_none_needed
// CHECK-NOT: noisy.reduce_noise
func.func @test_insert_noise_reduction_ops_add_none_needed() -> i5 {
  %0 = arith.constant 3 : i5
  %1 = arith.constant 4 : i5
  %2 = noisy.encode %0 : i5 -> !noisy.i32
  %3 = noisy.encode %1 : i5 -> !noisy.i32
  %4 = noisy.add %2, %3 : !noisy.i32
  %5 = noisy.add %4, %4 : !noisy.i32
  %6 = noisy.add %5, %5 : !noisy.i32
  %7 = noisy.add %6, %6 : !noisy.i32
  %8 = noisy.decode %7 : !noisy.i32 -> i5
  return %8 : i5
}


// CHECK-LABEL: test_add_after_mul
// CHECK: noisy.mul
// CHECK: noisy.reduce_noise
// CHECK: noisy.add
// CHECK: noisy.add
// CHECK: noisy.add
// CHECK: noisy.add
// CHECK: noisy.decode
// CHECK: return
func.func @test_add_after_mul() -> i5 {
  %0 = arith.constant 3 : i5
  %1 = arith.constant 4 : i5
  %2 = noisy.encode %0 : i5 -> !noisy.i32
  %3 = noisy.encode %1 : i5 -> !noisy.i32
  // Noise: 12
  %4 = noisy.mul %2, %3 : !noisy.i32
  // Noise: 24
  %5 = noisy.add %4, %3 : !noisy.i32
  // Noise: 25
  %6 = noisy.add %5, %5 : !noisy.i32
  // Noise: 26
  %7 = noisy.add %6, %6 : !noisy.i32
  // Noise: 27
  %8 = noisy.add %7, %7 : !noisy.i32
  %9 = noisy.decode %8 : !noisy.i32 -> i5
  return %9 : i5
}

// This test checks that the solver can find a single insertion point
// for a reduce_noise op that handles two branches, each of which would
// also need a reduce_noise op if handled separately.
// CHECK-LABEL: test_single_insertion_branching
// CHECK: noisy.mul
// CHECK-NOT: noisy.add
// CHECK-COUNT-1: noisy.reduce_noise
// CHECK-NOT: noisy.reduce_noise
func.func @test_single_insertion_branching() -> i5 {
  %0 = arith.constant 3 : i5
  %1 = arith.constant 4 : i5
  %2 = noisy.encode %0 : i5 -> !noisy.i32
  %3 = noisy.encode %1 : i5 -> !noisy.i32
  // Noise: 12
  %4 = noisy.mul %2, %3 : !noisy.i32
  // Noise: 24

  // branch 1
  %b1 = noisy.add %4, %3 : !noisy.i32
  // Noise: 25
  %b2 = noisy.add %b1, %3 : !noisy.i32
  // Noise: 25
  %b3 = noisy.add %b2, %3 : !noisy.i32
  // Noise: 26
  %b4 = noisy.add %b3, %3 : !noisy.i32
  // Noise: 27

  // branch 2
  %c1 = noisy.sub %4, %2 : !noisy.i32
  // Noise: 25
  %c2 = noisy.sub %c1, %3 : !noisy.i32
  // Noise: 25
  %c3 = noisy.sub %c2, %3 : !noisy.i32
  // Noise: 26
  %c4 = noisy.sub %c3, %3 : !noisy.i32
  // Noise: 27

  %x1 = noisy.decode %b4 : !noisy.i32 -> i5
  %x2 = noisy.decode %c4 : !noisy.i32 -> i5
  %x3 = arith.addi %x1, %x2 : i5
  return %x3 : i5
}

// same as test_single_insertion_branching, but because the last two values
// are multiplied, we need two reduce_noise ops, one on each branch.
// CHECK-LABEL: test_double_insertion_branching
// CHECK: noisy.mul
// CHECK: noisy.add
// CHECK-COUNT-2: noisy.reduce_noise
// CHECK-NOT: noisy.reduce_noise
// CHECK: noisy.mul
func.func @test_double_insertion_branching() -> i5 {
  %0 = arith.constant 3 : i5
  %1 = arith.constant 4 : i5
  %2 = noisy.encode %0 : i5 -> !noisy.i32
  %3 = noisy.encode %1 : i5 -> !noisy.i32
  // Noise: 12
  %4 = noisy.mul %2, %3 : !noisy.i32
  // Noise: 24

  // branch 1
  %b1 = noisy.add %4, %3 : !noisy.i32
  // Noise: 25
  %b2 = noisy.add %b1, %3 : !noisy.i32
  // Noise: 25
  %b3 = noisy.add %b2, %3 : !noisy.i32
  // Noise: 26
  %b4 = noisy.add %b3, %3 : !noisy.i32
  // Noise: 27

  // branch 2
  %c1 = noisy.add %4, %2 : !noisy.i32
  // Noise: 25
  %c2 = noisy.add %c1, %3 : !noisy.i32
  // Noise: 25
  %c3 = noisy.add %c2, %3 : !noisy.i32
  // Noise: 26
  %c4 = noisy.add %c3, %3 : !noisy.i32
  // Noise: 27

  %exit = noisy.mul %b4, %c4 : !noisy.i32

  %x1 = noisy.decode %exit : !noisy.i32 -> i5
  return %x1 : i5
}
