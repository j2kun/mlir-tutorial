// RUN: tutorial-opt %s > %t
// RUN FileCheck %s < %t

module {
  // CHECK-LABEL: test_type_syntax
  func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.poly
    return %arg0 : !poly.poly<10>
  }

  // CHECK-LABEL: test_op_syntax
  func.func @test_op_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.add
    %0 = poly.add %arg0, %arg1 : !poly.poly<10>
    // CHECK: poly.sub
    %1 = poly.sub %arg0, %arg1 : !poly.poly<10>
    // CHECK: poly.mul
    %2 = poly.mul %arg0, %arg1 : !poly.poly<10>

    %3 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    // CHECK: poly.from_tensor
    %4 = poly.from_tensor %3 : tensor<3xi32> -> !poly.poly<10>

    %5 = arith.constant 7 : i32
    // CHECK: poly.eval
    %6 = poly.eval %4, %5 : (!poly.poly<10>, i32) -> i32

    %z = complex.constant [1.0, 2.0] : complex<f64>
    // CHECK: poly.eval
    %complex_eval = poly.eval %4, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>

    %7 = tensor.from_elements %arg0, %arg1 : tensor<2x!poly.poly<10>>
    // CHECK: poly.add
    %8 = poly.add %7, %7 : tensor<2x!poly.poly<10>>

    // CHECK: poly.constant
    %10 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
    %11 = poly.constant dense<[2, 3, 4]> : tensor<3xi8> : !poly.poly<10>
    %12 = poly.constant dense<"0x020304"> : tensor<3xi8> : !poly.poly<10>
    %13 = poly.constant dense<4> : tensor<100xi32> : !poly.poly<10>

    // CHECK: poly.to_tensor
    %14 = poly.to_tensor %1 : !poly.poly<10> -> tensor<10xi32>

    return %4 : !poly.poly<10>
  }
}
