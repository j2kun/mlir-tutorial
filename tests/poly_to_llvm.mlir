// RUN: tutorial-opt --poly-to-llvm %s | FileCheck %s

// CHECK-NOT: poly
func.func @test_poly_fn(%arg : i32) -> i32 {
  %tens = tensor.splat %arg : tensor<10xi32>
  %input = poly.from_tensor %tens : tensor<10xi32> -> !poly.poly<10>
  %0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  %1 = poly.add %0, %input : !poly.poly<10>
  %2 = poly.mul %1, %1 : !poly.poly<10>
  %3 = poly.sub %2, %input : !poly.poly<10>
  %4 = poly.eval %3, %arg: (!poly.poly<10>, i32) -> i32
  return %4 : i32
}
